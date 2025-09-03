#!/usr/bin/env python3
"""
M2-BERT Ray Actor System
Async, non-blocking architecture for long-context document processing
Using Ray actors for distributed, stateful computation
"""

import ray
import torch
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
import time
import os
from m2bert_compatibility import M2BertModel, M2BertConfig, load_pretrained_m2bert


@dataclass
class DocumentChunk:
    """Represents a chunk of a document for processing"""
    doc_id: str
    chunk_id: int
    text: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Results from M2-BERT processing"""
    doc_id: str
    chunk_id: int
    embeddings: np.ndarray
    attention_scores: Optional[np.ndarray]
    processing_time: float
    metadata: Dict[str, Any]


@ray.remote
class TokenizerActor:
    """Actor for tokenization - handles text preprocessing"""
    
    def __init__(self, model_path: str, max_length: int = 32768):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        print(f"TokenizerActor initialized with max_length={max_length}")
    
    async def tokenize(self, chunks: List[DocumentChunk]) -> List[Dict]:
        """Tokenize a batch of document chunks"""
        texts = [chunk.text for chunk in chunks]
        
        # Batch tokenization
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Return tokenized data with metadata
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                'doc_id': chunk.doc_id,
                'chunk_id': chunk.chunk_id,
                'input_ids': encoded['input_ids'][i],
                'attention_mask': encoded['attention_mask'][i],
                'metadata': chunk.metadata
            })
        
        return results
    
    async def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
class ModelActor:
    """Actor for M2-BERT model - handles inference"""
    
    def __init__(self, model_path: str):
        print(f"Initializing ModelActor with model from {model_path}")
        
        # Load model
        self.model, self.config = load_pretrained_m2bert(model_path)
        
        # Move to appropriate device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Compile model for better performance
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="max-autotune")
        
        print(f"ModelActor ready on {self.device}")
    
    async def process_batch(self, tokenized_batch: List[Dict]) -> List[ProcessingResult]:
        """Process a batch of tokenized inputs"""
        start_time = time.perf_counter()
        
        # Stack inputs
        input_ids = torch.stack([item['input_ids'] for item in tokenized_batch]).to(self.device)
        attention_mask = torch.stack([item['attention_mask'] for item in tokenized_batch]).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # Extract embeddings
        hidden_states = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']
        
        # Create results
        results = []
        for i, item in enumerate(tokenized_batch):
            # Mean pooling over sequence
            mask = attention_mask[i].unsqueeze(-1).expand(hidden_states[i].size())
            masked_hidden = hidden_states[i] * mask
            summed = torch.sum(masked_hidden, dim=0)
            count = torch.clamp(mask.sum(dim=0), min=1e-9)
            mean_pooled = (summed / count).cpu().numpy()
            
            results.append(ProcessingResult(
                doc_id=item['doc_id'],
                chunk_id=item['chunk_id'],
                embeddings=mean_pooled,
                attention_scores=None,  # Could extract if needed
                processing_time=time.perf_counter() - start_time,
                metadata=item['metadata']
            ))
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            'device': str(self.device),
            'hidden_size': self.config.hidden_size,
            'max_positions': self.config.max_position_embeddings,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'monarch_blocks': self.config.monarch_mlp_nblocks
        }


@ray.remote
class DocumentActor:
    """Actor for document management - handles chunking and reassembly"""
    
    def __init__(self, chunk_size: int = 30000, overlap: int = 1000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents = {}
        print(f"DocumentActor initialized with chunk_size={chunk_size}")
    
    async def chunk_document(self, doc_id: str, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Split a document into overlapping chunks"""
        chunks = []
        
        # Store original document
        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata or {},
            'chunks': []
        }
        
        # Create chunks
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=len(chunks),
                text=chunk_text,
                metadata={
                    'start_pos': i,
                    'end_pos': min(i + self.chunk_size, len(text)),
                    'total_chunks': 0  # Will update
                }
            )
            chunks.append(chunk)
        
        # Update total chunks
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        self.documents[doc_id]['chunks'] = chunks
        return chunks
    
    async def reassemble_results(self, results: List[ProcessingResult]) -> Dict:
        """Reassemble chunked results into document-level output"""
        # Group by document
        doc_results = {}
        for result in results:
            if result.doc_id not in doc_results:
                doc_results[result.doc_id] = []
            doc_results[result.doc_id].append(result)
        
        # Combine embeddings for each document
        final_results = {}
        for doc_id, doc_chunks in doc_results.items():
            # Sort by chunk ID
            doc_chunks.sort(key=lambda x: x.chunk_id)
            
            # Average embeddings (or could concatenate)
            embeddings = np.stack([chunk.embeddings for chunk in doc_chunks])
            doc_embedding = np.mean(embeddings, axis=0)
            
            final_results[doc_id] = {
                'embedding': doc_embedding,
                'num_chunks': len(doc_chunks),
                'total_time': sum(chunk.processing_time for chunk in doc_chunks),
                'metadata': self.documents.get(doc_id, {}).get('metadata', {})
            }
        
        return final_results


@ray.remote
class OrchestratorActor:
    """Main orchestrator - coordinates the pipeline"""
    
    def __init__(self, model_path: str, num_model_replicas: int = 2):
        self.model_path = model_path
        self.num_model_replicas = num_model_replicas
        
        # Initialize actors
        self.tokenizer = TokenizerActor.remote(model_path)
        self.document_actor = DocumentActor.remote()
        
        # Create multiple model actors for parallelism
        self.model_actors = [
            ModelActor.remote(model_path) 
            for _ in range(num_model_replicas)
        ]
        
        self.current_model_idx = 0
        print(f"OrchestratorActor initialized with {num_model_replicas} model replicas")
    
    def get_next_model_actor(self):
        """Round-robin model actor selection"""
        actor = self.model_actors[self.current_model_idx]
        self.current_model_idx = (self.current_model_idx + 1) % self.num_model_replicas
        return actor
    
    async def process_document(self, doc_id: str, text: str, metadata: Dict = None) -> Dict:
        """Process a complete document through the pipeline"""
        start_time = time.perf_counter()
        
        # Step 1: Chunk the document
        chunks = await self.document_actor.chunk_document.remote(doc_id, text, metadata)
        print(f"Document {doc_id} split into {len(chunks)} chunks")
        
        # Step 2: Tokenize chunks in batches
        batch_size = 4
        tokenized_batches = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            tokenized = await self.tokenizer.tokenize.remote(batch)
            tokenized_batches.append(tokenized)
        
        # Step 3: Process through model actors (parallel)
        result_futures = []
        for tokenized_batch in tokenized_batches:
            model_actor = self.get_next_model_actor()
            future = model_actor.process_batch.remote(tokenized_batch)
            result_futures.append(future)
        
        # Wait for all results
        all_results = []
        for future in result_futures:
            batch_results = await future
            all_results.extend(batch_results)
        
        # Step 4: Reassemble results
        final_results = await self.document_actor.reassemble_results.remote(all_results)
        
        # Add timing
        total_time = time.perf_counter() - start_time
        final_results[doc_id]['pipeline_time'] = total_time
        
        print(f"Document {doc_id} processed in {total_time:.2f}s")
        return final_results
    
    async def process_documents(self, documents: List[Tuple[str, str, Dict]]) -> Dict:
        """Process multiple documents concurrently"""
        tasks = []
        for doc_id, text, metadata in documents:
            task = self.process_document(doc_id, text, metadata)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine all results
        combined = {}
        for result in results:
            combined.update(result)
        
        return combined


async def demo_pipeline():
    """Demonstrate the Ray actor pipeline"""
    print("="*70)
    print("M2-BERT RAY ACTOR PIPELINE DEMO")
    print("="*70)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    model_path = "./m2_models/models--togethercomputer--m2-bert-80M-32k-retrieval/snapshots/a2ccdc5b5661a282c77545e586a019f387ab7a48"
    
    # Create orchestrator
    orchestrator = OrchestratorActor.remote(model_path, num_model_replicas=2)
    
    # Test documents
    documents = [
        ("doc1", "Monarch matrices are a revolutionary approach to efficient transformers. " * 500, {"type": "research"}),
        ("doc2", "Python is a versatile programming language used in data science. " * 500, {"type": "tutorial"}),
        ("doc3", "The implementation of Ray actors enables distributed computing. " * 500, {"type": "technical"}),
    ]
    
    print(f"\nProcessing {len(documents)} documents...")
    
    # Process documents
    results = await orchestrator.process_documents.remote(documents)
    
    # Display results
    print("\nResults:")
    for doc_id, result in results.items():
        print(f"\n{doc_id}:")
        print(f"  Embedding shape: {result['embedding'].shape}")
        print(f"  Embedding norm: {np.linalg.norm(result['embedding']):.4f}")
        print(f"  Chunks processed: {result['num_chunks']}")
        print(f"  Total time: {result['pipeline_time']:.2f}s")
        print(f"  Metadata: {result['metadata']}")
    
    # Compute similarity between documents
    print("\nDocument Similarity (cosine):")
    doc_ids = list(results.keys())
    for i in range(len(doc_ids)):
        for j in range(i+1, len(doc_ids)):
            emb1 = results[doc_ids[i]]['embedding']
            emb2 = results[doc_ids[j]]['embedding']
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"  {doc_ids[i]} <-> {doc_ids[j]}: {similarity:.4f}")
    
    ray.shutdown()


if __name__ == "__main__":
    # Check if model exists
    model_path = "./m2_models/models--togethercomputer--m2-bert-80M-32k-retrieval/snapshots/a2ccdc5b5661a282c77545e586a019f387ab7a48"
    
    if not os.path.exists(model_path):
        print("Model not found. Please download first:")
        print("  python -c \"from huggingface_hub import snapshot_download; snapshot_download('togethercomputer/m2-bert-80M-32k-retrieval', cache_dir='./m2_models')\"")
    else:
        # Run async demo
        asyncio.run(demo_pipeline())