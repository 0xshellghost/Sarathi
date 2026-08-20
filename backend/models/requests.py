"""Pydantic request models with strict validation."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """User's raw problem description for the AI pipeline."""

    user_input: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Plain-text description of the user's legal problem.",
        examples=["My landlord hasn't returned my security deposit of 50000 rupees after I vacated 3 months ago."],
    )
    session_id: str | None = Field(
        default=None,
        description="Existing session ID for multi-turn context. Omit to start a new session.",
    )


class FinalizeRequest(BaseModel):
    """Submitted form data for case finalization and PDF payload generation."""

    session_id: str = Field(..., description="Session ID from the analyze step.")
    case_type: str = Field(..., description="Legal domain classification.")
    extracted_data: dict = Field(
        ...,
        description="Key-value pairs matching the domain's entity schema.",
    )
