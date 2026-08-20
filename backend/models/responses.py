"""Pydantic response models and SSE event schemas."""

from datetime import datetime
from pydantic import BaseModel, Field


# ── PDF Payload (consumed by frontend for rendering) ─────────────


class PartyDetails(BaseModel):
    name: str
    address: str | None = None


class PDFPayload(BaseModel):
    """Structured payload the frontend uses to render a legal document PDF."""

    document_title: str
    case_type: str
    plaintiff_details: PartyDetails | None = None
    defendant_details: PartyDetails | None = None
    facts_of_case: str | None = None
    legal_clauses_invoked: list[str] = Field(default_factory=list)
    relief_sought: str | None = None
    additional_fields: dict | None = None


# ── API Responses ────────────────────────────────────────────────


class FinalizeResponse(BaseModel):
    status: str = "success"
    case_id: str
    session_id: str
    pdf_payload: PDFPayload


class CaseSummary(BaseModel):
    case_id: str
    session_id: str
    type: str
    summary: str
    created_at: datetime
    status: str


class CaseHistoryResponse(BaseModel):
    cases: list[CaseSummary]


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None


# ── SSE Event Schemas ────────────────────────────────────────────
# These match the frontend's expected event format in lib/stream.ts


class TokenEvent(BaseModel):
    """Streamed token for the AI's explanation."""

    type: str = "token"
    text: str


class FormFieldSchema(BaseModel):
    key: str
    label: str
    type: str = "text"
    required: bool = True


class FormSchema(BaseModel):
    case_type: str
    fields: list[FormFieldSchema]


class FormRequestEvent(BaseModel):
    """Sent after the explanation — tells the frontend to render a form."""

    type: str = "form_request"
    schema_: FormSchema = Field(..., alias="schema")

    model_config = {"populate_by_name": True}


class ErrorEvent(BaseModel):
    type: str = "error"
    message: str
    recoverable: bool = True


class CompleteEvent(BaseModel):
    type: str = "complete"
    session_id: str
