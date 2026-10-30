"""Native Chat Completions streaming helpers.

Raw stream chunks are OpenAI ``ChatCompletionChunk`` objects (the native
streaming result of the openai library / any OpenAI-compatible endpoint). This
module aggregates them into a normalized :class:`ChatResponse` and can
synthesize a single native chunk from a non-streaming response so that custom
providers without streaming support still produce a uniform raw event.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from loong_agent.types.messages import ToolCall, Usage

_FALLBACK_CHUNK_ID = "chatcmpl-loong-fallback"


def chunk_from_chat_response(response: Any) -> ChatCompletionChunk:
    """Synthesize one native chunk from a non-streaming :class:`ChatResponse`."""
    tool_calls: List[ChoiceDeltaToolCall] = []
    for index, tc in enumerate(response.tool_calls or []):
        tool_calls.append(
            ChoiceDeltaToolCall(
                index=index,
                id=tc.id or f"call_{index}",
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tc.name,
                    arguments=json.dumps(tc.arguments, ensure_ascii=False)
                    if isinstance(tc.arguments, dict)
                    else str(tc.arguments),
                ),
            )
        )

    delta = ChoiceDelta(
        role="assistant",
        content=response.text or None,
        tool_calls=tool_calls or None,
    )
    choice = Choice(delta=delta, finish_reason=response.finish_reason, index=0)
    usage = CompletionUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
    return ChatCompletionChunk(
        id=_FALLBACK_CHUNK_ID,
        choices=[choice],
        created=int(time.time()),
        model=response.model or "unknown",
        object="chat.completion.chunk",
        usage=usage,
    )


class ChatCompletionAggregator:
    """Aggregate native ``ChatCompletionChunk`` deltas into a ``ChatResponse``.

    Imported lazily by callers to avoid a hard circular import; the aggregate
    mirrors :class:`loong_agent.llm.base.ChatResponse` fields including the
    vendor-extended ``reasoning`` text.
    """

    def __init__(self) -> None:
        self._id: Optional[str] = None
        self._model: Optional[str] = None
        self._text: List[str] = []
        self._reasoning: List[str] = []
        self._tool_calls: Dict[int, Dict[str, Any]] = {}
        self._finish_reason: Optional[str] = None
        self._usage = Usage()

    def add(self, chunk: ChatCompletionChunk) -> None:
        if self._id is None and getattr(chunk, "id", None):
            self._id = chunk.id
        if self._model is None and getattr(chunk, "model", None):
            self._model = chunk.model

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage = Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )

        for choice in chunk.choices or []:
            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                self._text.append(content)

            # Vendor extensions: deepseek-style ``reasoning_content`` or ``reasoning``.
            reasoning = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning:
                self._reasoning.append(reasoning)

            for tc in getattr(delta, "tool_calls", None) or []:
                index = tc.index
                entry = self._tool_calls.setdefault(
                    index, {"id": "", "name": "", "arguments": []}
                )
                if getattr(tc, "id", None):
                    entry["id"] = tc.id
                function = getattr(tc, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        entry["name"] += function.name
                    args = getattr(function, "arguments", None)
                    if args:
                        entry["arguments"].append(args)

            if choice.finish_reason:
                self._finish_reason = choice.finish_reason

    def finish(self) -> Any:
        """Build the normalized response (returns ``ChatResponse``)."""
        from loong_agent.llm.base import ChatResponse

        tool_calls: List[ToolCall] = []
        for index in sorted(self._tool_calls):
            entry = self._tool_calls[index]
            raw_args = "".join(entry["arguments"]) or "{}"
            try:
                args: Dict[str, Any] = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {"_raw": args}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(
                    id=entry["id"] or f"call_{index}",
                    name=entry["name"],
                    arguments=args,
                )
            )

        return ChatResponse(
            text="".join(self._text),
            tool_calls=tool_calls,
            usage=self._usage,
            model=self._model,
            finish_reason=self._finish_reason,
            reasoning="".join(self._reasoning),
            raw=self._id,
        )


__all__ = ["ChatCompletionAggregator", "chunk_from_chat_response"]
