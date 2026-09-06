from pydantic import BaseModel, SecretStr


class ModelConnector(BaseModel):
    base_url: str
    api_key: SecretStr | None


class ModelParameters(BaseModel):
    model_name: str
    temperature: float
    max_output_tokens: int
    max_context_tokens: int
    max_iterations: int
    use_streaming: bool
