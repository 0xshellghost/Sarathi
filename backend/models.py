from typing import Any
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    user_input: str = Field(..., description="The user's raw problem description")

class FinalizeRequest(BaseModel):
    case_type: str
    extracted_data: dict[str, Any]

class FormField(BaseModel):
    key: str
    label: str
    type: str
    required: bool = True

class FormSchema(BaseModel):
    case_type: str
    fields: list[FormField]

class FormRequestEvent(BaseModel):
    type: str = "form_request"
    schema: FormSchema

class PartyDetails(BaseModel):
    name: str
    address: str | None = None

class PDFPayload(BaseModel):
    document_title: str
    plaintiff_details: PartyDetails | None = None
    defendant_details: PartyDetails | None = None
    facts_of_case: str | None = None
    legal_clauses_invoked: list[str] | None = None
    relief_sought: str | None = None

class FinalizeResponse(BaseModel):
    status: str
    case_id: str
    pdf_payload: PDFPayload

class CaseSummary(BaseModel):
    case_id: str
    type: str
    summary: str
    created_at: str
    status: str

class CaseHistoryResponse(BaseModel):
    cases: list[CaseSummary]