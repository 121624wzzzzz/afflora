"""Strict multi-turn chat templating and assistant-only labels."""

from __future__ import annotations

import re
from typing import Any


_WS_RE = re.compile(r"\s+")
SUPPORTED_ROLES = {"user", "assistant"}


class ConversationFormatError(ValueError):
    """Raised when a row is not a complete alternating conversation."""


def canonical_text(value: str) -> str:
    return _WS_RE.sub(" ", value).strip()


def messages_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("conversations")
    if not isinstance(raw, list) or not raw:
        raise ConversationFormatError("missing_nonempty_conversations")

    messages: list[dict[str, str]] = []
    for index, turn in enumerate(raw):
        if not isinstance(turn, dict):
            raise ConversationFormatError(f"turn_{index}_not_object")
        role = turn.get("role")
        if role not in SUPPORTED_ROLES:
            raise ConversationFormatError(f"turn_{index}_unsupported_role:{role}")
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ConversationFormatError(f"turn_{index}_empty_content")
        if "<|im_start|>" in content or "<|im_end|>" in content:
            raise ConversationFormatError(f"turn_{index}_contains_chatml_control_token")
        messages.append({"role": role, "content": content})

    roles = [message["role"] for message in messages]
    if roles[0] != "user":
        raise ConversationFormatError("conversation_does_not_start_with_user")
    if roles[-1] != "assistant":
        raise ConversationFormatError("conversation_does_not_end_with_assistant")
    if any(left == right for left, right in zip(roles, roles[1:])):
        raise ConversationFormatError("roles_do_not_strictly_alternate")
    return messages


def first_user_key(row: dict[str, Any]) -> str:
    return canonical_text(messages_from_row(row)[0]["content"])


def assistant_turn_count(row: dict[str, Any]) -> int:
    return sum(message["role"] == "assistant" for message in messages_from_row(row))


def _find_subsequence(values: list[int], pattern: list[int], start: int) -> int:
    stop = len(values) - len(pattern) + 1
    for index in range(start, max(start, stop)):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def tokenize_conversation(
    row: dict[str, Any], tokenizer: Any, max_seq_len: int
) -> dict[str, list[int]]:
    """Render with the native template and label assistant content plus turn end."""

    messages = messages_from_row(row)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        enable_thinking=False,
    )
    input_ids = list(input_ids)

    assistant_start = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    turn_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if not assistant_start or not turn_end:
        raise RuntimeError("Tokenizer does not expose the expected ChatML markers.")

    labels = [-100] * len(input_ids)
    cursor = 0
    found = 0
    while True:
        start = _find_subsequence(input_ids, assistant_start, cursor)
        if start < 0:
            break
        content_start = start + len(assistant_start)
        end = _find_subsequence(input_ids, turn_end, content_start)
        if end < 0:
            raise RuntimeError("Assistant turn is missing <|im_end|> after templating.")
        supervised_end = end + len(turn_end)
        labels[content_start:supervised_end] = input_ids[content_start:supervised_end]
        found += 1
        cursor = supervised_end

    expected = sum(message["role"] == "assistant" for message in messages)
    if found != expected:
        raise RuntimeError(
            f"Assistant marker mismatch after templating: expected={expected}, found={found}."
        )

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    if not any(label != -100 for label in labels):
        raise RuntimeError("Truncation removed every supervised assistant token.")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

