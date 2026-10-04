from pydantic import BaseModel, Field

from agent_orchestrator.domain.model.model_listing import ModelListing


class ModelSchema(BaseModel):
    name: str = Field(description="The value to send as model_name.")
    context_tokens: int
    max_output_tokens: int
    thinking: bool = Field(description="Whether the model reasons before answering.")


class ModelListSchema(BaseModel):
    models: list[ModelSchema]
    pinned_steps: dict[str, str] = Field(
        default_factory=dict,
        description="Steps that always use one model, whatever model_name asks for.",
    )

    @classmethod
    def from_domain(cls, listing: ModelListing) -> ModelListSchema:
        return cls(
            models=[
                ModelSchema(
                    name=model.name,
                    context_tokens=model.context_tokens,
                    max_output_tokens=model.max_output_tokens,
                    thinking=model.thinking,
                )
                for model in listing.models
            ],
            pinned_steps={step.value: model for step, model in listing.pinned_steps.items()},
        )
