from typing import Literal

DEFAULT_THINK_TAGS = ("<think>", "</think>")
PLACEHOLDER_ASSISTANT_CONTENT = (
    "(placeholder - never trained on directly; see utils/strings.py:PLACEHOLDER_ASSISTANT_CONTENT)"
) #ChatMLCollator needs a trailing assistant entry to slice off

## WILDJAILBREAK (safety refusal / abstention)
WILDJAILBREAK_DATA_TYPES = (
    "vanilla_benign",
    "vanilla_harmful",
    "adversarial_benign",
    "adversarial_harmful",
)


## CALCULATOR
OFFICIAL_REPOSITORY = "https://github.com/google-deepmind/mathematics_dataset"
OFFICIAL_COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
OFFICIAL_MAX_QUESTION_LENGTH = 160
OFFICIAL_MAX_ANSWER_LENGTH = 30

DEFAULT_MODULES = (
    "arithmetic__add_or_sub",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul_div_multiple",
    "arithmetic__mixed",
)

CALCULATOR_PREFIX = "CALCULATOR:"

CALCULATOR_ROLLOUT_COUNT_NAMES = (
    "calculator_total",
    "correct",
    "calculator",
)

DEFAULT_METADATA_KEYS = (
    "eval_example_index",
    "example_id",
    "source",
    "gold_answer",
    "data_type",
    "should_abstain",
    "prompt_style",
)


## OPD CONFIG (opd/opd_config.py field types)
LossType = Literal["seq_rkl", "tok_rkl", "tok_fkl", "tok_js", "tok_skl"]
LossReduction = Literal["trajectory_mean"]
TrainingAlgorithm = Literal["opd", "sft"]


## CHAT-TEMPLATE TURN MARKERS (opd/eval_rollouts.py: recover the request from a rendered prompt)
QWEN_USER_START = "<|im_start|>user\n"
QWEN_MESSAGE_END = "<|im_end|>"
GEMMA_USER_START = "<start_of_turn>user\n"
GEMMA_MESSAGE_END = "<end_of_turn>"

# User-turn markers per chat family, keyed by a lowercase tokenizer-name substring.
CHAT_TURN_MARKERS: dict[str, tuple[str, str]] = {
    "qwen": (QWEN_USER_START, QWEN_MESSAGE_END),
    "gemma": (GEMMA_USER_START, GEMMA_MESSAGE_END),
}
# Gemma's turn tokens lack the <|...|> shape, so they are listed explicitly.
GEMMA_TURN_TOKENS = ("<start_of_turn>", GEMMA_MESSAGE_END)
REQUEST_BOUNDARY = "\n\nRequest: "
