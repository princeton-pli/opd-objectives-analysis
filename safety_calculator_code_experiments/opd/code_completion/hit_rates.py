from __future__ import annotations

INSTANCE_FORMS = frozenset({".Int128"})


def name_closes(rest: str, call_suffix: str, continuation: str = "") -> bool:
    if call_suffix:
        if continuation in INSTANCE_FORMS and rest.startswith("()"):
            rest = rest[2:]
        return rest.startswith(call_suffix[0])
    return rest == "" or rest[0] in "\n("


def classify(completion: str, continuation: str, call_suffix: str) -> bool:
    rest = completion[len(continuation):]
    return completion.startswith(continuation) and name_closes(rest, call_suffix, continuation)
