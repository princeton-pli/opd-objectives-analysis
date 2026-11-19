from __future__ import annotations

from collections.abc import Sequence

from utils import DEFAULT_THINK_TAGS


def prompt_prefills_thinking(
    text: str,
    *,
    think_tags: tuple[str, str] = DEFAULT_THINK_TAGS,
) -> bool:
    """Whether the rendered generation prompt ends inside a think block."""
    think_start, _ = think_tags
    return bool(think_start and text.rstrip().endswith(think_start))


def extract_final_response(
    text: str,
    *,
    think_tags: tuple[str, str] = DEFAULT_THINK_TAGS,
    thinking_prefilled: bool = False,
) -> str | None:
    stripped = text.strip()
    think_start, think_end = think_tags

    if think_end and think_end in stripped:
        return stripped.rsplit(think_end, 1)[1].strip()
    if thinking_prefilled or (think_start and think_start in stripped):
        return None
    return stripped


def truncate_token_ids_at_stop(
    token_ids: Sequence[int],
    stop_token_ids: int | Sequence[int] | None,
) -> list[int]:
    if stop_token_ids is None:
        return list(token_ids)

    if isinstance(stop_token_ids, int):
        stops = {stop_token_ids}
    else:
        stops = set(stop_token_ids)

    for index, token_id in enumerate(token_ids):
        if token_id in stops:
            return list(token_ids[:index])
    return list(token_ids)
