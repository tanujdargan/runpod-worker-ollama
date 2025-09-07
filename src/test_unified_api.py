#!/usr/bin/env python3
"""
Test script for the unified Pear Care API
Tests all major endpoints and functionality
"""

import asyncio
import json
import aiohttp
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class APITester:
    """Test suite for unified API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def run_test(self, name: str, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Running test: {name}")
        start_time = time.time()
        
        try:
            result = await test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"✅ {name} passed ({duration:.2f}s)")
                self.test_results.append({"name": name, "status": "passed", "duration": duration})
            else:
                print(f"❌ {name} failed ({duration:.2f}s)")
                self.test_results.append({"name": name, "status": "failed", "duration": duration})
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {name} errored: {e} ({duration:.2f}s)")
            self.test_results.append({"name": name, "status": "error", "duration": duration, "error": str(e)})
    
    async def test_health_check(self) -> bool:
        """Test health check endpoint"""
        async with self.session.get(f"{self.base_url}/health") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("status") == "healthy"
            return False
    
    async def test_models_endpoint(self) -> bool:
        """Test models endpoint"""
        async with self.session.get(f"{self.base_url}/models") as response:
            if response.status == 200:
                data = await response.json()
                return "data" in data and len(data["data"]) > 0
            return False
    
    async def test_chat_completion(self) -> bool:
        """Test chat completion endpoint"""
        payload = {
            "model": "phraser",
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                return "choices" in data and len(data["choices"]) > 0
            return False
    
    async def test_streaming_chat(self) -> bool:
        """Test streaming chat completion"""
        payload = {
            "model": "phraser",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            "max_tokens": 50,
            "temperature": 0.1,
            "stream": True
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                chunk_count = 0
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith("data: "):
                            chunk_count += 1
                            if chunk_count >= 3:  # Got some chunks
                                return True
            return False
    
    async def test_langchain_consultation_start(self) -> bool:
        """Test starting Langchain consultation"""
        payload = {
            "symptoms": "I have a severe headache and nausea",
            "patient_data": {
                "age": 30,
                "gender": "female",
                "pregnant": False
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with self.session.post(
            f"{self.base_url}/v1/langchain/consultation",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                return "session_id" in data and "status" in data
            return False
    
    async def test_langchain_streaming(self) -> bool:
        """Test Langchain consultation streaming"""
        # First start a consultation
        payload = {
            "symptoms": "I have a mild headache",
            "patient_data": {
                "age": 25,
                "gender": "male"
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with self.session.post(
            f"{self.base_url}/v1/langchain/consultation",
            json=payload,
            headers=headers
        ) as response:
            if response.status != 200:
                return False
            
            data = await response.json()
            session_id = data.get("session_id")
            
            if not session_id:
                return False
        
        # Now test streaming
        async with self.session.get(
            f"{self.base_url}/v1/langchain/consultation/{session_id}/stream",
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status == 200:
                update_count = 0
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith("data: "):
                            update_count += 1
                            if update_count >= 2:  # Got some updates
                                return True
            return False
    
    async def test_text_completion(self) -> bool:
        """Test text completion endpoint"""
        payload = {
            "model": "phraser",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                return "choices" in data and len(data["choices"]) > 0
            return False
    
    async def test_chat_test_endpoint(self) -> bool:
        """Test the chat test endpoint"""
        async with self.session.post(f"{self.base_url}/v1/chat/test") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("status") == "success"
            return False
    
    async def test_langchain_test_endpoint(self) -> bool:
        """Test the Langchain test endpoint"""
        async with self.session.post(f"{self.base_url}/v1/langchain/test") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("status") == "success"
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Pear Care Unified API Test Suite")
        print(f"📍 Testing against: {self.base_url}")
        
        # Basic functionality tests
        await self.run_test("Health Check", self.test_health_check)
        await self.run_test("Models Endpoint", self.test_models_endpoint)
        
        # Chat completion tests
        await self.run_test("Chat Completion", self.test_chat_completion)
        await self.run_test("Streaming Chat", self.test_streaming_chat)
        await self.run_test("Text Completion", self.test_text_completion)
        
        # Langchain tests
        await self.run_test("Langchain Consultation Start", self.test_langchain_consultation_start)
        await self.run_test("Langchain Streaming", self.test_langchain_streaming)
        
        # Test endpoints
        await self.run_test("Chat Test Endpoint", self.test_chat_test_endpoint)
        await self.run_test("Langchain Test Endpoint", self.test_langchain_test_endpoint)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.test_results if r["status"] == "passed")
        failed = sum(1 for r in self.test_results if r["status"] == "failed")
        errors = sum(1 for r in self.test_results if r["status"] == "error")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        
        total_time = sum(r["duration"] for r in self.test_results)
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status_emoji = {"passed": "✅", "failed": "❌", "error": "💥"}
            emoji = status_emoji.get(result["status"], "❓")
            print(f"  {emoji} {result['name']}: {result['status']} ({result['duration']:.2f}s)")
            if "error" in result:
                print(f"      Error: {result['error']}")
        
        if passed == total:
            print("\n🎉 All tests passed! The API is working correctly.")
        else:
            print(f"\n⚠️  {failed + errors} test(s) failed. Check the logs for details.")

async def main():
    """Main test function"""
    async with APITester() as tester:
        await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
