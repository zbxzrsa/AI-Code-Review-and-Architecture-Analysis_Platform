"""
AI Model Streaming Response

Implements:
- Server-Sent Events (SSE) streaming
- Chunked response handling
- Latency reduction through streaming
- Progress tracking
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str
    chunk_index: int
    is_final: bool
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class AIStreamingClient:
    """
    Handles streaming responses from AI models.
    
    Supports OpenAI, Anthropic, and local model streaming.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        timeout: float = 60.0,
    ):
        self.provider = provider
        self.timeout = timeout
    
    async def stream_completion(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        on_chunk: Optional[Callable[[StreamChunk], Awaitable[None]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream completion from AI model.
        
        Yields chunks as they arrive from the API.
        """
        chunk_index = 0
        
        if self.provider == "openai":
            async for chunk in self._stream_openai(prompt, model, system_prompt, max_tokens, temperature):
                chunk.chunk_index = chunk_index
                chunk_index += 1
                
                if on_chunk:
                    await on_chunk(chunk)
                
                yield chunk
                
        elif self.provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt, model, system_prompt, max_tokens, temperature):
                chunk.chunk_index = chunk_index
                chunk_index += 1
                
                if on_chunk:
                    await on_chunk(chunk)
                
                yield chunk
        else:
            # Fallback to non-streaming
            result = await self._call_non_streaming(prompt, model, system_prompt, max_tokens, temperature)
            yield StreamChunk(
                content=result,
                chunk_index=0,
                is_final=True,
                timestamp=datetime.now(timezone.utc),
            )
    
    async def _stream_openai(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from OpenAI API."""
        try:
            import openai
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            client = openai.AsyncOpenAI()
            
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        chunk_index=0,
                        is_final=chunk.choices[0].finish_reason is not None,
                        timestamp=datetime.now(timezone.utc),
                        metadata={"model": model, "provider": "openai"},
                    )
                    
        except ImportError:
            logger.warning("OpenAI library not installed, using mock streaming")
            async for chunk in self._mock_stream(prompt):
                yield chunk
    
    async def _stream_anthropic(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic()
            
            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                system=system_prompt or "",
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(
                        content=text,
                        chunk_index=0,
                        is_final=False,
                        timestamp=datetime.now(timezone.utc),
                        metadata={"model": model, "provider": "anthropic"},
                    )
                
                # Final chunk
                yield StreamChunk(
                    content="",
                    chunk_index=0,
                    is_final=True,
                    timestamp=datetime.now(timezone.utc),
                )
                
        except ImportError:
            logger.warning("Anthropic library not installed, using mock streaming")
            async for chunk in self._mock_stream(prompt):
                yield chunk
    
    async def _mock_stream(self, prompt: str) -> AsyncIterator[StreamChunk]:
        """Mock streaming for testing."""
        response = f"Mock response for: {prompt[:50]}..."
        words = response.split()
        
        for i, word in enumerate(words):
            await asyncio.sleep(0.05)  # Simulate latency
            yield StreamChunk(
                content=word + " ",
                chunk_index=i,
                is_final=(i == len(words) - 1),
                timestamp=datetime.now(timezone.utc),
            )
    
    async def _call_non_streaming(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Fallback non-streaming call."""
        return f"Non-streaming response for: {prompt[:50]}..."


def create_sse_response(
    stream: AsyncIterator[StreamChunk],
    event_type: str = "message",
) -> StreamingResponse:
    """
    Create Server-Sent Events response from stream.
    
    Usage:
        @app.get("/stream")
        async def stream_endpoint():
            stream = ai_client.stream_completion(prompt)
            return create_sse_response(stream)
    """
    
    async def generate():
        try:
            async for chunk in stream:
                data = {
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "is_final": chunk.is_final,
                    "timestamp": chunk.timestamp.isoformat(),
                }
                
                if chunk.metadata:
                    data["metadata"] = chunk.metadata
                
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(data)}\n\n"
                
                if chunk.is_final:
                    yield f"event: done\n"
                    yield f"data: {{}}\n\n"
                    break
                    
        except Exception as e:
            error_data = {"error": str(e)}
            yield f"event: error\n"
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class StreamingCodeReview:
    """
    Streaming code review with progress updates.
    
    Provides real-time feedback during analysis.
    """
    
    def __init__(self, ai_client: AIStreamingClient):
        self.ai_client = ai_client
    
    async def review_with_progress(
        self,
        code: str,
        analysis_types: list[str],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream code review with progress updates.
        
        Yields:
            - progress: Analysis progress (0-100)
            - stage: Current analysis stage
            - findings: Detected issues as they're found
            - complete: Final summary when done
        """
        total_stages = len(analysis_types)
        findings = []
        
        for idx, analysis_type in enumerate(analysis_types):
            # Emit progress
            progress = int((idx / total_stages) * 100)
            yield {
                "type": "progress",
                "stage": analysis_type,
                "progress": progress,
                "message": f"Analyzing {analysis_type}...",
            }
            
            # Perform analysis with streaming
            prompt = self._build_analysis_prompt(code, analysis_type)
            
            full_response = ""
            async for chunk in self.ai_client.stream_completion(
                prompt=prompt,
                model="gpt-4-turbo",
            ):
                full_response += chunk.content
                
                # Emit partial findings if detected
                partial_findings = self._extract_findings(full_response, analysis_type)
                for finding in partial_findings:
                    if finding not in findings:
                        findings.append(finding)
                        yield {
                            "type": "finding",
                            "finding": finding,
                        }
            
            # Stage complete
            yield {
                "type": "stage_complete",
                "stage": analysis_type,
                "progress": int(((idx + 1) / total_stages) * 100),
            }
        
        # Final summary
        yield {
            "type": "complete",
            "progress": 100,
            "total_findings": len(findings),
            "findings": findings,
        }
    
    def _build_analysis_prompt(self, code: str, analysis_type: str) -> str:
        """Build analysis prompt for specific type."""
        prompts = {
            "security": f"Analyze this code for security vulnerabilities:\n\n{code}",
            "performance": f"Analyze this code for performance issues:\n\n{code}",
            "quality": f"Analyze this code for quality issues:\n\n{code}",
            "architecture": f"Analyze this code for architectural issues:\n\n{code}",
        }
        return prompts.get(analysis_type, f"Analyze this code:\n\n{code}")
    
    def _extract_findings(self, response: str, analysis_type: str) -> list:
        """Extract findings from partial response."""
        # Simple extraction - in production, use structured parsing
        findings = []
        
        lines = response.split("\n")
        for line in lines:
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                findings.append({
                    "type": analysis_type,
                    "description": line.strip()[2:],
                })
        
        return findings


def create_streaming_review_response(
    review_stream: AsyncIterator[Dict[str, Any]],
) -> StreamingResponse:
    """Create streaming response for code review."""
    
    async def generate():
        try:
            async for update in review_stream:
                yield f"data: {json.dumps(update)}\n\n"
                
                if update.get("type") == "complete":
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                    
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
