from .code_task import CodeTask
from .language_task import LanguageTask
from .math_task import MathTask
from .pie_cpp_task import PieCppTask


TASK_REGISTRY = {
    "math": MathTask,
    "language": LanguageTask,
    "code": CodeTask,
    "code_cpp": PieCppTask,
}


def create_task(task_type: str, dataset_config: dict, action_strategy: dict, tokenizer=None):
    task_cls = TASK_REGISTRY.get(task_type)
    if task_cls is None:
        raise ValueError(f"Unknown task type: {task_type}")
    return task_cls(dataset_config=dataset_config, action_strategy=action_strategy, tokenizer=tokenizer)
