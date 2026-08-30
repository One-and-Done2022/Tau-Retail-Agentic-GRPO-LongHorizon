# Copyright Sierra

from typing import Optional, Union
from tau_bench.envs.base import Env
from tau_bench.envs.user import UserStrategy


def get_env(
    env_name: str,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_split: str,
    user_provider: Optional[str] = None,
    user_api_base: Optional[str] = None,
    user_seed: Optional[int] = None,
    task_index: Optional[int] = None,
) -> Env:
    if env_name != "retail":
        raise ValueError(f"Only the retail environment is supported, got: {env_name}")

    from tau_bench.envs.retail import MockRetailDomainEnv

    return MockRetailDomainEnv(
        user_strategy=user_strategy,
        user_model=user_model,
        task_split=task_split,
        user_provider=user_provider,
        user_api_base=user_api_base,
        user_seed=user_seed,
        task_index=task_index,
    )
