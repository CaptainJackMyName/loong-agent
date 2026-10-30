"""Convert native Chat Completions stream chunks into Responses API stream events.

The provider layer streams ``openai.types.chat.ChatCompletionChunk`` objects (the
native Chat Completions streaming result). Consumers of ``agent.run_streamed()``
however expect the *Responses* event vocabulary used by the openai-agents SDK:
``response.created``, ``response.output_item.added``, ``response.output_text.delta``,
``response.function_call_arguments.delta`` and ``response.completed``.

This module bridges the two: :class:`ChatCompletionToResponsesConverter` is a
per-turn stateful mapper that folds chunks in via :meth:`add` and closes the
stream via :meth:`finish`, emitting Responses events
(``openai.types.responses.ResponseStreamEvent``) that mirror the semantics of
openai-agents' ``ChatCmplStreamHandler`` — deliberately simplified to the output
shapes loong-agent actually supports (text, vendor reasoning fields, tool calls).

Supported mapping:

- first chunk                 -> ``response.created`` (``Response``, in_progress)
- first text delta            -> ``response.output_item.added`` (assistant message)
                                 + ``response.content_part.added``
- text deltas                 -> ``response.output_text.delta``
- ``reasoning_content`` delta -> reasoning item + ``response.reasoning_summary_text.delta``
- ``reasoning`` delta         -> reasoning item + ``response.reasoning_text.delta``
- tool call delta             -> ``response.output_item.added`` (function call)
                                 + ``response.function_call_arguments.delta``
- stream end                  -> ``response.content_part.done`` / ``response.output_item.done``
                                 per open item + ``response.completed`` (usage included)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import Content, Summary
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as AddedEventPart,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    Part as DoneEventPart,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

logger = logging.getLogger(__name__)

FAKE_RESPONSES_ID = "__fake_id__"
"""Placeholder item id, mirroring openai-agents' fake id for synthesized objects."""


class _SequenceNumber:
    """Monotonic per-stream event counter (Responses events carry their own)."""

    def __init__(self) -> None:
        self._value = 0

    def get_and_increment(self) -> int:
        value = self._value
        self._value += 1
        return value


@dataclass
class _BufferedFunctionCall:
    """Accumulates one streamed Chat Completions function tool call."""

    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    opened: bool = False
    output_index: int = 0


@dataclass
class _ConverterState:
    """Tracks which output items have been announced to stream consumers."""

    started: bool = False
    response_id: str | None = None
    model: str | None = None
    created_at: float = field(default_factory=time.time)
    usage: CompletionUsage | None = None

    reasoning_item: ResponseReasoningItem | None = None
    reasoning_done: bool = False
    active_summary_index: int | None = None

    assistant_message: ResponseOutputMessage | None = None
    text_part: ResponseOutputText | None = None

    function_calls: dict[int, _BufferedFunctionCall] = field(default_factory=dict)

    # Output items in the order they were announced to consumers.
    outputs: list[ResponseOutputItem] = field(default_factory=list)

    warned_unsupported_choice: bool = False

    def register(self, item: ResponseOutputItem) -> int:
        """Record an announced output item; return its output_index."""
        self.outputs.append(item)
        return len(self.outputs) - 1


def _output_text(text: str = "") -> ResponseOutputText:
    return ResponseOutputText(
        text=text,
        type="output_text",
        annotations=[],
        logprobs=[],
    )


def _build_response_usage(usage: CompletionUsage | None) -> ResponseUsage | None:
    """Convert a Chat Completions usage payload into a Responses usage payload."""
    if usage is None:
        return None

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", None) or 0
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) or 0

    return ResponseUsage(
        input_tokens=usage.prompt_tokens or 0,
        input_tokens_details=InputTokensDetails(
            cached_tokens=cached_tokens,
            cache_write_tokens=0,
        ),
        output_tokens=usage.completion_tokens or 0,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=usage.total_tokens or 0,
    )


class ChatCompletionToResponsesConverter:
    """Per-turn stateful mapper: Chat Completions chunks -> Responses stream events."""

    def __init__(self) -> None:
        self._state = _ConverterState()
        self._sequence_number = _SequenceNumber()

    # -- public API ----------------------------------------------------------

    def add(self, chunk: ChatCompletionChunk) -> list[ResponseStreamEvent]:
        """Fold one chunk in; return the Responses events it produced (may be empty)."""
        state = self._state
        events: list[ResponseStreamEvent] = []

        if not state.started:
            state.started = True
            events.append(
                ResponseCreatedEvent(
                    response=self._initial_response(chunk),
                    type="response.created",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        if getattr(chunk, "usage", None) is not None:
            state.usage = chunk.usage

        choice = self._first_choice(chunk)
        if choice is None or choice.delta is None:
            return events

        delta = choice.delta

        self._add_reasoning_summary(delta, events)
        self._add_reasoning_text(delta, events)
        self._add_text(delta, events)
        self._add_tool_calls(delta, events)

        # Close an open reasoning summary part once non-reasoning output starts,
        # mirroring the Responses API boundary between reasoning and answer.
        if (
            state.active_summary_index is not None
            and (delta.content is not None or delta.tool_calls)
        ):
            self._finish_reasoning_summary_part(events)

        return events

    def finish(self) -> list[ResponseStreamEvent]:
        """Close all open items and emit the terminal ``response.completed`` event."""
        state = self._state
        events: list[ResponseStreamEvent] = []

        if not state.started:
            # Empty stream (nothing ever arrived): still emit a minimal, valid
            # created + completed pair so consumers see a well-formed stream.
            state.started = True
            events.append(
                ResponseCreatedEvent(
                    response=self._initial_response(None),
                    type="response.created",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        self._finish_reasoning_summary_part(events)
        self._finish_reasoning_item(events)
        self._finish_assistant_message(events)
        self._finish_function_calls(events)

        final_response = Response(
            id=state.response_id or FAKE_RESPONSES_ID,
            created_at=state.created_at,
            model=state.model or "unknown",
            object="response",
            output=list(state.outputs),
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
            status="completed",
            usage=_build_response_usage(state.usage),
        )
        events.append(
            ResponseCompletedEvent(
                response=final_response,
                type="response.completed",
                sequence_number=self._sequence_number.get_and_increment(),
            )
        )
        return events

    # -- chunk handling ------------------------------------------------------

    def _initial_response(self, chunk: ChatCompletionChunk | None) -> Response:
        """Build the in-progress Response object announced by ``response.created``."""
        state = self._state
        if chunk is not None:
            if state.response_id is None and getattr(chunk, "id", None):
                state.response_id = chunk.id
            if state.model is None and getattr(chunk, "model", None):
                state.model = chunk.model
        return Response(
            id=state.response_id or FAKE_RESPONSES_ID,
            created_at=state.created_at,
            model=state.model or "unknown",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
            status="in_progress",
        )

    def _first_choice(self, chunk: ChatCompletionChunk) -> Choice | None:
        choices = chunk.choices or []
        for choice in choices:
            if choice.index == 0:
                return choice

        if choices and not self._state.warned_unsupported_choice:
            self._state.warned_unsupported_choice = True
            logger.warning(
                "Chat Completions streaming with nonzero choice indexes is not "
                "supported; only choice index 0 will be processed."
            )
        return None

    def _add_text(self, delta: Any, events: list[ResponseStreamEvent]) -> None:
        content = getattr(delta, "content", None)
        if content is None:
            return

        state = self._state
        # Drop a single leading empty delta: it opens a spurious empty text part.
        if state.text_part is None and content == "":
            return

        if state.assistant_message is None:
            state.assistant_message = ResponseOutputMessage(
                id=FAKE_RESPONSES_ID,
                content=[],
                role="assistant",
                type="message",
                status="in_progress",
            )
            output_index = state.register(state.assistant_message)
            state.text_part = _output_text()
            events.append(
                ResponseOutputItemAddedEvent(
                    item=state.assistant_message,
                    output_index=output_index,
                    type="response.output_item.added",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )
            events.append(
                ResponseContentPartAddedEvent(
                    content_index=0,
                    item_id=FAKE_RESPONSES_ID,
                    output_index=output_index,
                    part=_output_text(),
                    type="response.content_part.added",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        output_index = state.outputs.index(state.assistant_message)
        events.append(
            ResponseTextDeltaEvent(
                content_index=0,
                delta=content,
                item_id=FAKE_RESPONSES_ID,
                output_index=output_index,
                type="response.output_text.delta",
                sequence_number=self._sequence_number.get_and_increment(),
                logprobs=[],
            )
        )
        state.text_part.text += content

    def _add_reasoning_summary(self, delta: Any, events: list[ResponseStreamEvent]) -> None:
        """Map deepseek-style ``reasoning_content`` deltas onto a reasoning summary."""
        reasoning = getattr(delta, "reasoning_content", None)
        if not reasoning:
            return

        state = self._state
        if state.reasoning_item is None:
            state.reasoning_item = ResponseReasoningItem(
                id=FAKE_RESPONSES_ID,
                summary=[],
                type="reasoning",
            )
            output_index = state.register(state.reasoning_item)
            events.append(
                ResponseOutputItemAddedEvent(
                    item=state.reasoning_item,
                    output_index=output_index,
                    type="response.output_item.added",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        output_index = state.outputs.index(state.reasoning_item)
        if state.active_summary_index is None:
            summary_index = len(state.reasoning_item.summary)
            state.reasoning_item.summary.append(Summary(text="", type="summary_text"))
            state.active_summary_index = summary_index
            events.append(
                ResponseReasoningSummaryPartAddedEvent(
                    item_id=FAKE_RESPONSES_ID,
                    output_index=output_index,
                    summary_index=summary_index,
                    part=AddedEventPart(text="", type="summary_text"),
                    type="response.reasoning_summary_part.added",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        summary_index = state.active_summary_index
        events.append(
            ResponseReasoningSummaryTextDeltaEvent(
                delta=reasoning,
                item_id=FAKE_RESPONSES_ID,
                output_index=output_index,
                summary_index=summary_index,
                type="response.reasoning_summary_text.delta",
                sequence_number=self._sequence_number.get_and_increment(),
            )
        )
        state.reasoning_item.summary[summary_index] = Summary(
            text=state.reasoning_item.summary[summary_index].text + reasoning,
            type="summary_text",
        )

    def _add_reasoning_text(self, delta: Any, events: list[ResponseStreamEvent]) -> None:
        """Map the generic vendor ``reasoning`` field onto reasoning text deltas."""
        reasoning = getattr(delta, "reasoning", None)
        if not reasoning or not isinstance(reasoning, str):
            return

        state = self._state
        if state.reasoning_item is None:
            state.reasoning_item = ResponseReasoningItem(
                id=FAKE_RESPONSES_ID,
                summary=[],
                content=[Content(text="", type="reasoning_text")],
                type="reasoning",
            )
            output_index = state.register(state.reasoning_item)
            events.append(
                ResponseOutputItemAddedEvent(
                    item=state.reasoning_item,
                    output_index=output_index,
                    type="response.output_item.added",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        output_index = state.outputs.index(state.reasoning_item)
        events.append(
            ResponseReasoningTextDeltaEvent(
                delta=reasoning,
                item_id=FAKE_RESPONSES_ID,
                output_index=output_index,
                content_index=0,
                type="response.reasoning_text.delta",
                sequence_number=self._sequence_number.get_and_increment(),
            )
        )
        if state.reasoning_item.content:
            accumulated = state.reasoning_item.content[0].text + reasoning
            state.reasoning_item.content[0] = Content(text=accumulated, type="reasoning_text")
        else:
            state.reasoning_item.content = [Content(text=reasoning, type="reasoning_text")]

    def _add_tool_calls(self, delta: Any, events: list[ResponseStreamEvent]) -> None:
        tool_call_deltas: list[ChoiceDeltaToolCall] = getattr(delta, "tool_calls", None) or []
        if not tool_call_deltas:
            return

        state = self._state
        for tc_delta in tool_call_deltas:
            buffered = state.function_calls.setdefault(
                tc_delta.index, _BufferedFunctionCall(index=tc_delta.index)
            )

            if getattr(tc_delta, "id", None):
                buffered.call_id = tc_delta.id
            function = getattr(tc_delta, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    buffered.name = function.name
                if getattr(function, "arguments", None):
                    buffered.arguments += function.arguments

            # Open the item as soon as identity is known, then stream arguments.
            if not buffered.opened and buffered.name and buffered.call_id:
                item = self._function_call_item(buffered, arguments="")
                output_index = state.register(item)
                buffered.opened = True
                buffered.output_index = output_index
                events.append(
                    ResponseOutputItemAddedEvent(
                        item=item,
                        output_index=output_index,
                        type="response.output_item.added",
                        sequence_number=self._sequence_number.get_and_increment(),
                    )
                )

            if buffered.opened and function is not None and function.arguments:
                events.append(
                    ResponseFunctionCallArgumentsDeltaEvent(
                        delta=function.arguments,
                        item_id=FAKE_RESPONSES_ID,
                        output_index=buffered.output_index,
                        type="response.function_call_arguments.delta",
                        sequence_number=self._sequence_number.get_and_increment(),
                    )
                )

    # -- terminal handling ---------------------------------------------------

    def _finish_reasoning_summary_part(self, events: list[ResponseStreamEvent]) -> None:
        state = self._state
        if state.reasoning_item is None or state.active_summary_index is None:
            return

        summary_index = state.active_summary_index
        summary_text = state.reasoning_item.summary[summary_index].text
        events.append(
            ResponseReasoningSummaryPartDoneEvent(
                item_id=FAKE_RESPONSES_ID,
                output_index=state.outputs.index(state.reasoning_item),
                summary_index=summary_index,
                part=DoneEventPart(text=summary_text, type="summary_text"),
                type="response.reasoning_summary_part.done",
                sequence_number=self._sequence_number.get_and_increment(),
            )
        )
        state.active_summary_index = None

    def _finish_reasoning_item(self, events: list[ResponseStreamEvent]) -> None:
        state = self._state
        if state.reasoning_item is None or state.reasoning_done:
            return

        reasoning_item = state.reasoning_item
        output_index = state.outputs.index(reasoning_item)

        content = getattr(reasoning_item, "content", None)
        if content:
            events.append(
                ResponseReasoningTextDoneEvent(
                    item_id=FAKE_RESPONSES_ID,
                    output_index=output_index,
                    content_index=0,
                    text=content[0].text,
                    type="response.reasoning_text.done",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )

        events.append(
            ResponseOutputItemDoneEvent(
                item=reasoning_item,
                output_index=output_index,
                type="response.output_item.done",
                sequence_number=self._sequence_number.get_and_increment(),
            )
        )
        state.reasoning_done = True

    def _finish_assistant_message(self, events: list[ResponseStreamEvent]) -> None:
        state = self._state
        if state.assistant_message is None:
            return

        output_index = state.outputs.index(state.assistant_message)

        if state.text_part is not None:
            events.append(
                ResponseContentPartDoneEvent(
                    content_index=0,
                    item_id=FAKE_RESPONSES_ID,
                    output_index=output_index,
                    part=state.text_part,
                    type="response.content_part.done",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )
            state.assistant_message.content.append(state.text_part)

        state.assistant_message.status = "completed"
        events.append(
            ResponseOutputItemDoneEvent(
                item=state.assistant_message,
                output_index=output_index,
                type="response.output_item.done",
                sequence_number=self._sequence_number.get_and_increment(),
            )
        )

    def _finish_function_calls(self, events: list[ResponseStreamEvent]) -> None:
        state = self._state

        for index in sorted(state.function_calls):
            buffered = state.function_calls[index]
            call_id = buffered.call_id or f"call_{index}"
            name = buffered.name or "unknown"

            if not buffered.opened:
                # Never streamed (name/call_id arrived late or not at all):
                # emit the whole call at once, like openai-agents' fallback.
                buffered.call_id = call_id
                buffered.name = name
                item = self._function_call_item(buffered, arguments=buffered.arguments)
                output_index = state.register(item)
                buffered.opened = True
                buffered.output_index = output_index
                events.append(
                    ResponseOutputItemAddedEvent(
                        item=item,
                        output_index=output_index,
                        type="response.output_item.added",
                        sequence_number=self._sequence_number.get_and_increment(),
                    )
                )
                if buffered.arguments:
                    events.append(
                        ResponseFunctionCallArgumentsDeltaEvent(
                            delta=buffered.arguments,
                            item_id=FAKE_RESPONSES_ID,
                            output_index=output_index,
                            type="response.function_call_arguments.delta",
                            sequence_number=self._sequence_number.get_and_increment(),
                        )
                    )

            done_item = self._function_call_item(buffered, arguments=buffered.arguments)
            events.append(
                ResponseOutputItemDoneEvent(
                    item=done_item,
                    output_index=buffered.output_index,
                    type="response.output_item.done",
                    sequence_number=self._sequence_number.get_and_increment(),
                )
            )
            # Keep the registered output list in sync with the final item.
            state.outputs[buffered.output_index] = done_item

    @staticmethod
    def _function_call_item(
        buffered: _BufferedFunctionCall, arguments: str
    ) -> ResponseFunctionToolCall:
        return ResponseFunctionToolCall(
            id=FAKE_RESPONSES_ID,
            call_id=buffered.call_id,
            arguments=arguments,
            name=buffered.name,
            type="function_call",
        )


__all__ = [
    "ChatCompletionToResponsesConverter",
    "FAKE_RESPONSES_ID",
]
