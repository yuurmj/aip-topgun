__all__ = ["DogFightEnv"]


def __getattr__(name):
    """Avoid importing the training environment in lightweight inference."""

    if name == "DogFightEnv":
        from .envs.single_agent_env import DogFightEnv

        return DogFightEnv
    raise AttributeError(name)
