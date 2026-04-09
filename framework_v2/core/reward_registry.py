from .code_executor import CodeExecutor
from .rm_engine import RMEngine


REWARD_REGISTRY = {
    "model": RMEngine,
    "code_execution": CodeExecutor,
}


def create_reward_engine(reward_type: str, hardware_config: dict, dataset_config: dict = None):
    reward_cls = REWARD_REGISTRY.get(reward_type)
    if reward_cls is None:
        raise ValueError(f"Unknown reward type: {reward_type}")

    if reward_type == "code_execution":
        # language: prefer explicit hardware_config override, then dataset_config, then default
        language = (
            hardware_config.get("code_language")
            or (dataset_config or {}).get("language", "python")
        )
        return reward_cls(
            timeout_secs=hardware_config.get("code_timeout_secs", 10.0),
            n_timing_runs=hardware_config.get("code_timing_runs", 3),
            language=language,
        )

    return reward_cls(
        model_name=hardware_config["rm_name"],
        quantization=hardware_config.get("rm_quantization", False),
        max_batch_size=hardware_config["rm_max_batch_size"],
    )
