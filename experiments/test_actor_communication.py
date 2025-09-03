#!/usr/bin/env python
"""
Test Actor Communication
Validates actor-to-actor direct communication with PyTorch tensors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import ray
import torch
import time
from typing import Any, Optional

from thea_code_system.core.communication import (
    CommunicatingActor,
    CollaborativeActor,
    ActorRegistry,
    Message,
    PeerToPeerPool
)
from thea_code_system.core.enhanced_pool import EnhancedActor
from thea_code_system.core.scalars import ScalarOperations


class TestCommunicatingActor(EnhancedActor, CommunicatingActor):
    """Actor that can communicate with peers"""
    
    def __init__(self, actor_id: str, config=None):
        EnhancedActor.__init__(self, actor_id, config)
        CommunicatingActor.__init__(self)
        self.received_messages = []
        
    async def initialize(self) -> None:
        """Initialize actor"""
        await EnhancedActor.initialize(self)
        self.scalar_ops = ScalarOperations(self.device)
        
    async def process(self, data: Any) -> Any:
        """Process data"""
        if isinstance(data, (int, float)):
            tensor = self.scalar_ops.scalar(data)
            result = self.scalar_ops.mul(tensor, 3)
            return result.item()
        return data
    
    async def handle_message(self, message: Message) -> None:
        """Handle received message"""
        self.received_messages.append(message)
        print(f"Actor {self.actor_id} received: {message.content} from {message.sender_id}")
    
    def set_actor_id(self, actor_id: str) -> None:
        """Set actor ID"""
        self.actor_id = actor_id
    
    def get_received_count(self) -> int:
        """Get count of received messages"""
        return len(self.received_messages)


class TestCollaborativeActor(CollaborativeActor, EnhancedActor):
    """Actor that collaborates on computations"""
    
    def __init__(self, actor_id: str = "collaborative", config=None):
        EnhancedActor.__init__(self, actor_id, config)
        CollaborativeActor.__init__(self)
        
    async def initialize(self) -> None:
        """Initialize actor"""
        await EnhancedActor.initialize(self)
        
    async def process(self, data: torch.Tensor) -> Any:
        """Process with potential peer collaboration"""
        # Local computation
        local_result = data.mean()
        
        # Could request peer computation here
        return local_result.item() if hasattr(local_result, 'item') else local_result
    
    def set_actor_id(self, actor_id: str) -> None:
        """Set actor ID"""
        self.actor_id = actor_id


async def test_basic_communication():
    """Test basic actor-to-actor communication"""
    print("\n📊 Test 1: Basic Communication")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create registry as Ray actor
        registry = ActorRegistry.remote()
        
        # Create communicating actors
        Actor1 = ray.remote(TestCommunicatingActor)
        Actor2 = ray.remote(TestCommunicatingActor)
        
        actor1 = Actor1.remote("actor_1", None)
        actor2 = Actor2.remote("actor_2", None)
        
        # Initialize actors
        ray.get([
            actor1.initialize.remote(),
            actor2.initialize.remote()
        ])
        
        # Setup communication
        ray.get([
            actor1.setup_communication.remote(registry),
            actor2.setup_communication.remote(registry)
        ])
        
        # Register actors manually (normally done in setup_communication)
        ray.get([
            registry.register.remote("actor_1", actor1),
            registry.register.remote("actor_2", actor2)
        ])
        
        print("✅ Actors created and registered")
        
        # Send message from actor1 to actor2
        success = ray.get(
            actor1.send_message.remote("actor_2", "Hello from actor_1!")
        )
        
        # Give time for message to be processed
        await asyncio.sleep(0.5)
        
        # Check if message was received
        received_count = ray.get(actor2.get_received_count.remote())
        
        if received_count > 0:
            print(f"✅ Message sent and received (count: {received_count})")
        else:
            print("❌ Message not received")
            return False
        
        # Test tensor sharing via message
        tensor = torch.randn(5, 5)
        success = ray.get(
            actor1.send_message.remote("actor_2", "Sharing tensor", tensor)
        )
        
        await asyncio.sleep(0.5)
        
        new_count = ray.get(actor2.get_received_count.remote())
        if new_count > received_count:
            print("✅ Tensor message sent successfully")
        else:
            print("❌ Tensor message failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_broadcast():
    """Test broadcasting to multiple actors"""
    print("\n📊 Test 2: Broadcast Communication")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        registry = ActorRegistry.remote()
        
        # Create multiple actors
        actors = []
        for i in range(4):
            Actor = ray.remote(TestCommunicatingActor)
            actor = Actor.remote(f"actor_{i}", None)
            ray.get(actor.initialize.remote())
            ray.get(actor.setup_communication.remote(registry))
            ray.get(registry.register.remote(f"actor_{i}", actor))
            actors.append(actor)
        
        print(f"✅ Created {len(actors)} actors")
        
        # Broadcast from first actor
        broadcaster = actors[0]
        sent_count = ray.get(
            broadcaster.broadcast.remote("Broadcast message to all!")
        )
        
        print(f"✅ Broadcasted to {sent_count} actors")
        
        # Give time for processing
        await asyncio.sleep(0.5)
        
        # Check all actors received (except sender)
        success = True
        for i, actor in enumerate(actors[1:], 1):
            count = ray.get(actor.get_received_count.remote())
            if count > 0:
                print(f"   Actor {i}: Received {count} messages")
            else:
                print(f"   Actor {i}: No messages received")
                success = False
        
        return success
        
    except Exception as e:
        print(f"❌ Broadcast test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_collaborative_computation():
    """Test collaborative computation between actors"""
    print("\n📊 Test 3: Collaborative Computation")
    print("-" * 40)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        registry = ActorRegistry.remote()
        
        # Create collaborative actors
        Actor1 = ray.remote(TestCollaborativeActor)
        Actor2 = ray.remote(TestCollaborativeActor)
        
        actor1 = Actor1.remote("collab_1", None)
        actor2 = Actor2.remote("collab_2", None)
        
        ray.get([
            actor1.initialize.remote(),
            actor2.initialize.remote()
        ])
        
        ray.get([
            actor1.setup_communication.remote(registry),
            actor2.setup_communication.remote(registry)
        ])
        
        ray.get([
            registry.register.remote("collab_1", actor1),
            registry.register.remote("collab_2", actor2)
        ])
        
        print("✅ Collaborative actors created")
        
        # Test computation request
        test_tensor = torch.randn(10)
        
        # Actor1 requests computation from Actor2
        result = ray.get(
            actor1.request_computation.remote("collab_2", "sum", test_tensor)
        )
        
        if result is not None:
            expected = test_tensor.sum()
            if torch.allclose(torch.tensor(result), expected):
                print(f"✅ Collaborative computation correct: {result:.4f}")
            else:
                print(f"❌ Result mismatch: {result} != {expected}")
                return False
        else:
            print("❌ No result received from peer")
            return False
        
        # Test mean computation
        result = ray.get(
            actor1.request_computation.remote("collab_2", "mean", test_tensor)
        )
        
        if result is not None:
            expected = test_tensor.mean()
            if torch.allclose(torch.tensor(result), expected):
                print(f"✅ Mean computation correct: {result:.4f}")
            else:
                print(f"❌ Mean mismatch: {result} != {expected}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Collaborative computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all communication tests"""
    print("\n" + "="*60)
    print("🎯 ACTOR COMMUNICATION TEST")
    print("="*60)
    
    results = {}
    
    # Test 1: Basic communication
    results['Basic Communication'] = await test_basic_communication()
    
    # Test 2: Broadcast
    results['Broadcast'] = await test_broadcast()
    
    # Test 3: Collaborative computation
    results['Collaboration'] = await test_collaborative_computation()
    
    # Summary
    print("\n" + "="*60)
    print("📊 COMMUNICATION TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Actor communication working!")
        print("Successfully demonstrated:")
        print("  ✅ Direct actor-to-actor messaging")
        print("  ✅ Tensor sharing between actors")
        print("  ✅ Broadcast to multiple actors")
        print("  ✅ Collaborative computation requests")
    else:
        print("\n⚠️ Some communication tests failed")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)