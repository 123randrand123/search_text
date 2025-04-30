from pydantic import BaseModel, Field


class SuggestionResponse(BaseModel):
    replacement: str = Field(
        ...,
        description=("Suggested replacement, similar meaning, avoids blocked terms."),
    )


class ValidationResponse(BaseModel):
    should_block: bool = Field(
        ..., description="True if suggestion inappropriate/violates policy."
    )
    reason: str = Field(..., description="Explanation if should_block is True.")
