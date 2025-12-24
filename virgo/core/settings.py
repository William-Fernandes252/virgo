from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type GenAIProvider = Literal["openai", "ollama"]
"""Supported GenAI providers."""


class VirgoSettings(BaseSettings):
    """Settings for the Virgo application."""

    model_config = SettingsConfigDict(env_prefix="virgo_")

    genai_provider: Annotated[
        GenAIProvider,
        Field(
            json_schema_extra={
                "description": "The GenAI provider to use for language model interactions.",
                "examples": ["openai", "ollama"],
            },
        ),
    ] = "openai"
    model_name: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "The name of the model to use from the GenAI provider. It should correspond to a valid model name for the selected provider. Also, ensure that the model supports tool usage. It is recommended to use models with reliable reasoning capabilities.",
                "examples": ["gpt-4-turbo", "qwen-plus-7b-chat"],
            }
        ),
    ] = "gpt-4-turbo"
    max_iterations: Annotated[
        int,
        Field(
            json_schema_extra={
                "description": "The maximum number of iterations for the Virgo agent's reasoning process. This limits how many times the agent can loop through its reasoning steps before producing a final output.",
                "examples": [5, 10],
            },
        ),
    ] = 5


__all__ = [
    "VirgoSettings",
    "GenAIProvider",
]
