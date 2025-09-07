"""
Streaming manager for Server-Sent Events (SSE) and WebSocket responses
Handles application-level streaming to bypass RunPod limitations
"""

import json
import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any

class StreamingManager:
    """Manages streaming responses for all endpoints"""
    
    def __init__(self):
        self.active_streams = {}
        self.stream_counter = 0
    
    def generate_stream_id(self) -> str:
        """Generate unique stream ID"""
        self.stream_counter += 1
        return f"stream_{self.stream_counter}"
    
    async def stream_sse(
        self, 
        generator: AsyncGenerator[Dict, None],
        headers: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Convert generator to Server-Sent Events format
        
        Args:
            generator: Async generator yielding data dictionaries
            headers: Optional SSE headers
            
        Yields:
            SSE-formatted strings
        """
        # Send initial headers if provided
        if headers:
            for key, value in headers.items():
                yield f":{key}: {value}\n\n"
        
        try:
            async for data in generator:
                # Format as SSE
                if isinstance(data, dict):
                    yield f"data: {json.dumps(data)}\n\n"
                elif isinstance(data, str):
                    # Already formatted SSE data
                    yield data
                else:
                    # Convert to JSON
                    yield f"data: {json.dumps(str(data))}\n\n"
        
        except Exception as e:
            # Send error event
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "streaming_error"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        
        finally:
            # Send completion event
            yield "data: [DONE]\n\n"
    
    async def stream_chat_completion(
        self,
        generator: AsyncGenerator[Dict, None],
        model: str,
        stream_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion in OpenAI-compatible format
        
        Args:
            generator: Generator yielding chat completion chunks
            model: Model name for response metadata
            stream_id: Optional stream identifier
            
        Yields:
            SSE-formatted chat completion chunks
        """
        if not stream_id:
            stream_id = self.generate_stream_id()
        
        try:
            async for chunk in generator:
                # Ensure chunk has required OpenAI format
                if "id" not in chunk:
                    chunk["id"] = stream_id
                if "object" not in chunk:
                    chunk["object"] = "chat.completion.chunk"
                if "model" not in chunk:
                    chunk["model"] = model
                
                yield f"data: {json.dumps(chunk)}\n\n"
        
        except Exception as e:
            error_chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "model": model,
                "error": {
                    "message": str(e),
                    "type": "chat_completion_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        finally:
            yield "data: [DONE]\n\n"
    
    async def stream_langchain_consultation(
        self,
        generator: AsyncGenerator[Dict, None],
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream Langchain consultation updates
        
        Args:
            generator: Generator yielding consultation updates
            session_id: Session identifier
            
        Yields:
            SSE-formatted consultation updates
        """
        try:
            async for update in generator:
                # Add session metadata
                update["session_id"] = session_id
                update["timestamp"] = asyncio.get_event_loop().time()
                
                yield f"data: {json.dumps(update)}\n\n"
        
        except Exception as e:
            error_update = {
                "session_id": session_id,
                "error": {
                    "message": str(e),
                    "type": "consultation_error"
                },
                "timestamp": asyncio.get_event_loop().time()
            }
            yield f"data: {json.dumps(error_update)}\n\n"
        
        finally:
            completion_update = {
                "session_id": session_id,
                "status": "completed",
                "timestamp": asyncio.get_event_loop().time()
            }
            yield f"data: {json.dumps(completion_update)}\n\n"
    
    async def stream_agent_updates(
        self,
        agent_generator: AsyncGenerator[Dict, None],
        agent_name: str,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream individual agent updates with metadata
        
        Args:
            agent_generator: Generator from individual agent
            agent_name: Name of the agent
            session_id: Optional session ID
            
        Yields:
            SSE-formatted agent updates
        """
        try:
            async for update in agent_generator:
                # Add agent metadata
                formatted_update = {
                    "agent": agent_name,
                    "timestamp": asyncio.get_event_loop().time(),
                    **update
                }
                
                if session_id:
                    formatted_update["session_id"] = session_id
                
                yield f"data: {json.dumps(formatted_update)}\n\n"
        
        except Exception as e:
            error_update = {
                "agent": agent_name,
                "error": {
                    "message": str(e),
                    "type": "agent_error"
                },
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if session_id:
                error_update["session_id"] = session_id
            
            yield f"data: {json.dumps(error_update)}\n\n"
    
    async def multiplex_streams(
        self,
        *generators: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Multiplex multiple SSE streams into one
        
        Args:
            *generators: Multiple SSE generators to multiplex
            
        Yields:
            Combined SSE stream
        """
        # Convert generators to tasks
        tasks = []
        queues = []
        
        for i, generator in enumerate(generators):
            queue = asyncio.Queue()
            queues.append(queue)
            
            async def stream_to_queue(gen, q):
                try:
                    async for item in gen:
                        await q.put(item)
                except Exception as e:
                    await q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
                finally:
                    await q.put(None)  # Signal completion
            
            task = asyncio.create_task(stream_to_queue(generator, queue))
            tasks.append(task)
        
        # Read from all queues until all are done
        active_queues = set(range(len(queues)))
        
        while active_queues:
            # Wait for any queue to have data
            done_tasks = []
            
            for i in active_queues.copy():
                try:
                    item = await asyncio.wait_for(queues[i].get(), timeout=0.1)
                    if item is None:
                        active_queues.remove(i)
                    else:
                        yield item
                except asyncio.TimeoutError:
                    continue
            
            # Small delay to prevent busy waiting
            if active_queues:
                await asyncio.sleep(0.01)
        
        # Clean up tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def create_sse_headers(self) -> Dict[str, str]:
        """Create standard SSE headers"""
        return {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
