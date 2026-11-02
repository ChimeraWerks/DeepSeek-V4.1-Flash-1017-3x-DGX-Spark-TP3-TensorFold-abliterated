# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
from copy import deepcopy
from typing import Any

def coalesce_instruction_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one leading system message for strict Qwen/Ornith templates.

    OpenAI reasoning clients may send their primary instructions with the
    ``developer`` role and retain additional ``system`` history.  Ornith accepts
    neither ``developer`` nor more than one ``system`` message.  Preserve every
    instruction block in original order, promote them to system authority, and
    merge them into the one leading message that its template accepts.
    """

    instruction_messages: list[dict[str, Any]] = []
    non_system_messages: list[dict[str, Any]] = []
    for message in messages:
        copied = deepcopy(message)
        if copied.get("role") in _INSTRUCTION_ROLES:
            instruction_messages.append(copied)
        else:
            non_system_messages.append(copied)
    if not instruction_messages:
        return non_system_messages

    merged = instruction_messages[0]
    merged["role"] = "system"
    merged["content"] = "\n\n".join(
        text
        for text in (
            _system_content_text(message.get("content"))
            for message in instruction_messages
        )
        if text
    )
    return [merged, *non_system_messages]
