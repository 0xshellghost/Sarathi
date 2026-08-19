import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import (
    AnalyzeRequest, 
    FinalizeRequest, 
    FinalizeResponse,
    CaseHistoryResponse
)

app = FastAPI(title="Sarathi API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/sessions/history", response_model=CaseHistoryResponse)
async def get_history():
    return {
        "cases": [
            {
                "case_id": "case_mock_001",
                "type": "rent_deposit_dispute",
                "summary": "Recovery of deposit from Arun Sharma",
                "created_at": "2023-10-27T14:32:00Z",
                "status": "completed"
            }
        ]
    }

@app.get("/api/v1/sessions/cases/{case_id}")
async def get_case(case_id: str):
    return {
        "case_id": case_id,
        "pdf_payload": {
            "document_title": "Legal Notice for Recovery of Security Deposit",
            "defendant_details": {"name": "Arun Sharma"}
        }
    }

@app.post("/api/v1/action/finalize", response_model=FinalizeResponse)
async def finalize_case(data: FinalizeRequest):
    return {
        "status": "success",
        "case_id": "case_mock_002",
        "pdf_payload": {
            "document_title": "Legal Notice for Recovery of Security Deposit",
            "plaintiff_details": {"name": "The Plaintiff"},
            "defendant_details": {
                "name": data.extracted_data.get("landlord_name", "Unknown"),
                "address": data.extracted_data.get("property_address", "Unknown")
            },
            "facts_of_case": "You vacated the premises...",
            "legal_clauses_invoked": ["Section 12 of Model Tenancy Act"],
            "relief_sought": f"Return of INR {data.extracted_data.get('deposit_amount', 0)}"
        }
    }

@app.post("/api/v1/action/analyze")
async def analyze_problem(request: Request):
    async def event_generator():
        words = ["Under ", "the ", "Model ", "Tenancy ", "Act, ", "your ", "landlord ", "is ", "legally ", "obligated ", "to ", "return ", "your ", "deposit ", "within ", "30 ", "days."]
        
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'text': word})}\n\n"
            await asyncio.sleep(0.05)
            
        schema = {
            "type": "form_request", 
            "schema": {
                "case_type": "rent_deposit_dispute",
                "fields": [
                    {"key": "landlord_name", "label": "Landlord's Full Name", "type": "text", "required": True},
                    {"key": "property_address", "label": "Rented Property Address", "type": "text", "required": True},
                    {"key": "deposit_amount", "label": "Deposit Amount (INR)", "type": "number", "required": True}
                ]
            }
        }
        yield f"data: {json.dumps(schema)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "Sarathi AI Backend is running"}