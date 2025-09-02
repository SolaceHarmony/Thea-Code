#!/usr/bin/env python
"""
Actor Communication Module
Enables direct actor-to-actor communication with PyTorch tensors
"""

import ray
import torch
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass
import asyncio
import time

from .scalars import ScalarOperations


@dataclass
class Message:
    """Message structure for actor communication"""
    sender_id: str
    recipient_id: str
    content: Any
    tensor_data: Optional[torch.Tensor] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@ray.remote
class ActorRegistry:
    """
    Registry for tracking and communicating with actors.
    Enables actors to find and message each other.
    This is a Ray actor itself to enable shared state.
    """
    
    def __init__(self):
        self._actors = {}
        self._actor_handles = {}
        self._message_queue = []
        
    def register(self, actor_id: str, actor_handle: ray.actor.ActorHandle) -> None:
        """Register an actor in the registry"""
        self._actors[actor_id] = {
            'handle': actor_handle,
            'registered_at': time.time(),
            'last_seen': time.time()
        }
        self._actor_handles[actor_id] = actor_handle
        
    def unregister(self, actor_id: str) -> None:
        """Remove actor from registry"""
        self._actors.pop(actor_id, None)
        self._actor_handles.pop(actor_id, None)
        
    def get_actor(self, actor_id: str) -> Optional[ray.actor.ActorHandle]:
        """Get actor handle by ID"""
        return self._actor_handles.get(actor_id)
    
    def list_actors(self) -> List[str]:
        """List all registered actor IDs"""
        return list(self._actors.keys())
    
    def update_heartbeat(self, actor_id: str) -> None:
        """Update last seen timestamp for actor"""
        if actor_id in self._actors:
            self._actors[actor_id]['last_seen'] = time.time()


class CommunicatingActor:
    """
    Mixin class that adds communication capabilities to actors.
    All math operations use PyTorch scalars.
    """
    
    def __init__(self):
        self._inbox = asyncio.Queue()
        self._outbox = asyncio.Queue()
        self._peers = {}
        self._registry = None
        self.scalar_ops = None
        
    def setup_communication(self, registry: ActorRegistry) -> None:
        """Setup communication with registry"""
        self._registry = registry
        if hasattr(self, 'actor_id'):
            # If registry is a Ray actor, use .remote()
            try:
                ray.get(registry.register.remote(self.actor_id, ray.get_runtime_context().current_actor))
            except:
                # Fallback for non-Ray registry
                registry.register(self.actor_id, ray.get_runtime_context().current_actor)
            
    async def send_message(self, recipient_id: str, content: Any, tensor: Optional[torch.Tensor] = None) -> bool:
        """Send message to another actor"""
        if not self._registry:
            return False
            
        # Get actor from Ray registry
        recipient = ray.get(self._registry.get_actor.remote(recipient_id))
        if not recipient:
            return False
        
        message = Message(
            sender_id=getattr(self, 'actor_id', 'unknown'),
            recipient_id=recipient_id,
            content=content,
            tensor_data=tensor.cpu() if tensor is not None else None
        )
        
        # Send to recipient's receive method
        try:
            ray.get(recipient.receive_message.remote(message))
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self, message: Message) -> None:
        """Receive message from another actor"""
        # Store in inbox
        await self._inbox.put(message)
        
        # Process immediately if handler exists
        if hasattr(self, 'handle_message'):
            await self.handle_message(message)
    
    async def broadcast(self, content: Any, tensor: Optional[torch.Tensor] = None) -> int:
        """Broadcast message to all known actors"""
        if not self._registry:
            return 0
            
        sent_count = 0
        actor_list = ray.get(self._registry.list_actors.remote())
        for actor_id in actor_list:
            if actor_id != getattr(self, 'actor_id', None):
                if await self.send_message(actor_id, content, tensor):
                    sent_count += 1
                    
        return sent_count
    
    async def get_next_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Get next message from inbox"""
        try:
            if timeout:
                return await asyncio.wait_for(self._inbox.get(), timeout)
            else:
                return await self._inbox.get()
        except asyncio.TimeoutError:
            return None
    
    def has_messages(self) -> bool:
        """Check if there are pending messages"""
        return not self._inbox.empty()


class CollaborativeActor(CommunicatingActor):
    """
    Actor that can collaborate with peers on computations.
    Ensures all operations use PyTorch tensors.
    """
    
    def __init__(self):
        super().__init__()
        self._computation_results = {}
        self._pending_computations = {}
        
    async def request_computation(self, peer_id: str, operation: str, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Request computation from peer actor"""
        # Ensure data is tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
            
        # Create computation request
        request = {
            'operation': operation,
            'data': data.cpu(),  # CPU for serialization
            'request_id': f"{getattr(self, 'actor_id', 'unknown')}_{time.time()}"
        }
        
        # Send request
        success = await self.send_message(peer_id, request, data)
        if not success:
            return None
            
        # Store as pending
        self._pending_computations[request['request_id']] = time.time()
        
        # Wait for response (with timeout)
        timeout = 5.0
        start = time.time()
        
        while time.time() - start < timeout:
            if request['request_id'] in self._computation_results:
                result = self._computation_results.pop(request['request_id'])
                self._pending_computations.pop(request['request_id'], None)
                return result
            await asyncio.sleep(0.1)
            
        # Timeout
        self._pending_computations.pop(request['request_id'], None)
        return None
    
    async def handle_computation_request(self, message: Message) -> None:
        """Handle computation request from peer"""
        content = message.content
        
        if not isinstance(content, dict) or 'operation' not in content:
            return
            
        operation = content.get('operation')
        data = content.get('data')
        request_id = content.get('request_id')
        
        # Perform computation
        result = None
        if operation == 'sum':
            result = data.sum()
        elif operation == 'mean':
            result = data.mean()
        elif operation == 'multiply':
            if hasattr(self, 'scalar_ops'):
                result = self.scalar_ops.mul(data, 2)
            else:
                result = data * 2
        
        # Send response
        if result is not None:
            response = {
                'request_id': request_id,
                'result': result
            }
            await self.send_message(message.sender_id, response, result)
    
    async def handle_computation_response(self, message: Message) -> None:
        """Handle computation response from peer"""
        content = message.content
        
        if not isinstance(content, dict) or 'request_id' not in content:
            return
            
        request_id = content.get('request_id')
        result = content.get('result')
        
        # Store result
        if request_id in self._pending_computations:
            self._computation_results[request_id] = result
    
    async def handle_message(self, message: Message) -> None:
        """Route messages to appropriate handlers"""
        content = message.content
        
        if isinstance(content, dict):
            if 'operation' in content and 'data' in content:
                await self.handle_computation_request(message)
            elif 'request_id' in content and 'result' in content:
                await self.handle_computation_response(message)


class PeerToPeerPool:
    """
    Pool that enables peer-to-peer communication between actors.
    All actors can directly communicate without going through orchestrator.
    """
    
    def __init__(self, actor_class: type, num_actors: int):
        self.actor_class = actor_class
        self.num_actors = num_actors
        self.registry = ActorRegistry()
        self.actors = []
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize pool with communication-enabled actors"""
        if self._initialized:
            return
            
        # Create actors
        for i in range(self.num_actors):
            # Create Ray remote actor
            RemoteActor = ray.remote(self.actor_class)
            actor = RemoteActor.remote()
            
            # Set actor ID
            ray.get(actor.set_actor_id.remote(f"peer_{i}"))
            
            # Setup communication
            ray.get(actor.setup_communication.remote(self.registry))
            
            # Initialize if needed
            if hasattr(actor, 'initialize'):
                ray.get(actor.initialize.remote())
                
            self.actors.append(actor)
            
        self._initialized = True
        
    async def execute_collaborative_task(self, task: str, data: List[torch.Tensor]) -> List[Any]:
        """Execute task with actors collaborating"""
        results = []
        
        # Distribute data to actors
        for i, tensor in enumerate(data):
            actor = self.actors[i % len(self.actors)]
            
            # Actor processes its data
            result = ray.get(actor.process.remote(tensor))
            
            # Actor can request help from peers if needed
            if i > 0:
                peer = self.actors[(i-1) % len(self.actors)]
                peer_result = ray.get(
                    actor.request_computation.remote(f"peer_{(i-1) % len(self.actors)}", "mean", tensor)
                )
                if peer_result is not None:
                    result = {'local': result, 'peer': peer_result}
                    
            results.append(result)
            
        return results
    
    async def broadcast_tensor(self, tensor: torch.Tensor) -> int:
        """Broadcast tensor to all actors"""
        if not self.actors:
            return 0
            
        # Use first actor to broadcast
        sender = self.actors[0]
        count = ray.get(sender.broadcast.remote("shared_tensor", tensor))
        return count
    
    async def shutdown(self) -> None:
        """Shutdown pool"""
        for actor in self.actors:
            try:
                ray.kill(actor)
            except:
                pass
        self._initialized = False