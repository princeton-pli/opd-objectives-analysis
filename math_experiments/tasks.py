import os
import re
from typing import Dict, Any, List, Union
from datasets import load_dataset, load_from_disk

from utils import extract_answer_math, strip_string
from grader import math_equal


DATASET_REGISTRY = {
    "math500_inst": {
        "dataset": "HuggingFaceH4/MATH-500", "split": "test", "task_type": "math",
        "prompt_suffix": "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    },
    "mathtrain500_inst": {
        "dataset": os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "math_train500_eval.parquet"), "task_type": "math",
        "prompt_suffix": "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    },
}


class BaseTask:
    """Base class for all dataset tasks."""
        
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def load_data(self):
        """
        Loads data based on the config 'dataset' field.
        Handles HF Hub, JSONL, and Parquet automatically.
        """
        path = self.config.get("dataset")
        split = self.config.get("split", "test")
        
        print(f"  - Loading data from: {path} (split={split})")
        
        if path and os.path.exists(path):
            if path.endswith(".parquet"):
                return load_dataset("parquet", data_files=path, split='train')
            elif path.endswith(".jsonl") or path.endswith(".json"):
                return load_dataset("json", data_files=path)
            elif os.path.exists(f"{path}/{split}"):
                return load_from_disk(f"{path}/{split}")
            elif os.path.isdir(path):
                return load_from_disk(path)
        try:
            ds = load_dataset(path, split=split)
            #ds = ds.rename_column("prompt", "question")
            check_cols = ds[list(ds.keys())[0]].column_names if hasattr(ds, "keys") else ds.column_names
            if "prompt" in check_cols:
                ds = ds.rename_column("prompt", "question")
            return ds
        except Exception as e:
            print(f"    CRITICAL ERROR: Could not load dataset '{path}'. Details: {e}")
            return []
            
    def format_prompt(self, item: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """
        Returns:
            - A list of dicts (messages) for chat templating.
            - A string if the prompt is already formatted or raw completion is needed.
        """
        raise NotImplementedError

    def get_gold(self, item: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def grade(self, response: str, gold: Any, item: Dict[str, Any]):
        raise NotImplementedError


class MathTask(BaseTask):
    def format_prompt(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        content = item.get("problem", item.get("question", ""))
        content = content + self.config.get("prompt_suffix", "")
        return [{"role": "user", "content": content}]

    def get_gold(self, item: Dict[str, Any]) -> Any:
        answer = item.get("answer", "")
        solution = item.get("solution", "")
        if answer and 'boxed' in str(answer):
            return extract_answer_math(answer)
        if answer:
            return strip_string(str(answer))
        if solution and 'boxed' in str(solution):
            return extract_answer_math(solution)
        return extract_answer_math(solution)

    def grade(self, response: str, gold: Any, item: Dict[str, Any]):
        pred = extract_answer_math(response)
        score = math_equal(pred, gold, timeout=True)
        return {"is_correct": score}


def get_task(name_or_path: str) -> BaseTask:
    """
    Returns the MathTask of a dataset in DATASET_REGISTRY.
    """
    name_or_path = re.sub(r'_\d+$', '', name_or_path)
    config = DATASET_REGISTRY[name_or_path]

    return MathTask(config)
