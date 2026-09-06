"""OpenAI-compatible chat completions adapter.

Uses ``httpx`` directly against any ``/v1/chat/completions`` endpoint so the SDK
does not hard-depend on a specific vendor SDK. Works with OpenAI, local servers
(LLaMA.cpp, Ollama, vLLM), and managed providers exposing the same contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from loong_agent.llm.base import ChatResponse, LLMProvider, ToolDefinition
from loong_agent.types.messages import Message, ToolCall, Usage

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **extra_kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.extra_headers = extra_headers or {}
        self.extra_kwargs = extra_kwargs

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=timeout),
            headers={"Authorization": f"Bearer {api_key}", **self.extra_headers},
        )

    # -- LLMProvider ---------------------------------------------------------
    def get_model_name(self) -> str:
        return self.model

    def clone(self, model: Optional[str] = None) -> "OpenAICompatibleProvider":
        return OpenAICompatibleProvider(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model or self.model,
            timeout=self.timeout,
            temperature=self.temperature,
            extra_headers=dict(self.extra_headers),
            **self.extra_kwargs,
        )

    async def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
        }
        if tools:
            payload["tools"] = [self._to_openai_tool(t) for t in tools]
            payload["tool_choice"] = "auto"

        temperature = kwargs.pop("temperature", self.temperature)
        if temperature is not None:
            payload["temperature"] = temperature

        payload.update(self.extra_kwargs)
        payload.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        logger.debug("LLM request -> %s", url)
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data)

    # -- serialization helpers ----------------------------------------------
    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.type == "system":
                out.append({"role": "system", "content": msg.text or ""})
            elif msg.type == "assistant":
                item: Dict[str, Any] = {"role": "assistant", "content": msg.text or ""}
                if msg.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                out.append(item)
            elif msg.type == "user":
                content: Any = msg.text or ""
                if msg.tool_results:
                    tool_msgs = [
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                        }
                        for tr in msg.tool_results
                    ]
                    if content:
                        out.append({"role": "user", "content": content})
                    out.extend(tool_msgs)
                else:
                    out.append({"role": "user", "content": content})
            # result messages are terminal and not forwarded to the model.
        return out

    @staticmethod
    def _to_openai_tool(tool: ToolDefinition) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    def _parse_response(self, data: Dict[str, Any]) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage_raw = data.get("usage") or {}

        tool_calls: List[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args: Dict[str, Any] = {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        return ChatResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAICompatibleProvider"]
