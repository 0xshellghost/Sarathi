"""
Output Formatter — Step 4 of the Deterministic AI Pipeline

Compiles the results of the previous three steps (intent, RAG, entities) into:
1. A form_request SSE event (field schema for the frontend to render)
2. A PDFPayload (structured data for legal document generation)
"""

import logging
from models.domain import (
    LegalDomain,
    IntentResult,
    RAGResult,
    DOMAIN_FIELD_SCHEMAS,
)

logger = logging.getLogger("sarathi.pipeline.output")

# ── Document title templates per domain ──────────────────────────
_DOCUMENT_TITLES: dict[LegalDomain, str] = {
    LegalDomain.RENT_DEPOSIT_DISPUTE: "Legal Notice for Recovery of Security Deposit",
    LegalDomain.CONSUMER_COMPLAINT: "Consumer Complaint under Consumer Protection Act, 2019",
    LegalDomain.EMPLOYMENT_DISPUTE: "Legal Notice regarding Employment Grievance",
    LegalDomain.PROPERTY_DISPUTE: "Legal Notice regarding Property Dispute",
    LegalDomain.CHEQUE_BOUNCE: "Legal Notice under Section 138 of the Negotiable Instruments Act",
    LegalDomain.GENERAL_LEGAL_QUERY: "Legal Advisory Summary",
}


def build_form_schema(intent: IntentResult) -> dict:
    """
    Build the form_request SSE event payload.

    This tells the frontend what form fields to render based on the
    classified legal domain.
    """
    fields = DOMAIN_FIELD_SCHEMAS.get(intent.domain, [])
    return {
        "type": "form_request",
        "schema": {
            "case_type": intent.domain.value,
            "fields": [
                {
                    "key": f.key,
                    "label": f.label,
                    "type": f.field_type,
                    "required": f.required,
                }
                for f in fields
            ],
        },
    }


def build_pdf_payload(
    intent: IntentResult,
    rag_results: list[RAGResult],
    entities: dict,
) -> dict:
    """
    Compile the final PDF payload from all pipeline outputs.

    This is the structured data the frontend uses to render a legal document.
    """
    domain = intent.domain

    # Extract party details based on domain
    plaintiff = _extract_plaintiff(domain, entities)
    defendant = _extract_defendant(domain, entities)

    # Compile invoked legal clauses from RAG results
    clauses = [
        f"{r.act_name}, {r.section}" for r in rag_results
    ] if rag_results else ["Applicable provisions of Indian law"]

    # Build facts from entities
    facts = _compose_facts(domain, entities)

    # Relief sought
    relief = _compose_relief(domain, entities)

    return {
        "document_title": _DOCUMENT_TITLES.get(domain, "Legal Document"),
        "case_type": domain.value,
        "plaintiff_details": plaintiff,
        "defendant_details": defendant,
        "facts_of_case": facts,
        "legal_clauses_invoked": clauses,
        "relief_sought": relief,
        "additional_fields": {
            k: v for k, v in entities.items()
            if k not in _PARTY_FIELD_KEYS
        },
    }


# ── Private Helpers ──────────────────────────────────────────────

_PARTY_FIELD_KEYS = {
    "landlord_name", "tenant_name", "complainant_name", "opposite_party_name",
    "employee_name", "employer_name", "plaintiff_name", "defendant_name",
    "payee_name", "drawer_name",
}

_PLAINTIFF_KEYS: dict[LegalDomain, tuple[str, str | None]] = {
    LegalDomain.RENT_DEPOSIT_DISPUTE: ("tenant_name", None),
    LegalDomain.CONSUMER_COMPLAINT: ("complainant_name", None),
    LegalDomain.EMPLOYMENT_DISPUTE: ("employee_name", None),
    LegalDomain.PROPERTY_DISPUTE: ("plaintiff_name", None),
    LegalDomain.CHEQUE_BOUNCE: ("payee_name", None),
    LegalDomain.GENERAL_LEGAL_QUERY: ("query_subject", None),
}

_DEFENDANT_KEYS: dict[LegalDomain, tuple[str, str | None]] = {
    LegalDomain.RENT_DEPOSIT_DISPUTE: ("landlord_name", "property_address"),
    LegalDomain.CONSUMER_COMPLAINT: ("opposite_party_name", "opposite_party_address"),
    LegalDomain.EMPLOYMENT_DISPUTE: ("employer_name", "employer_address"),
    LegalDomain.PROPERTY_DISPUTE: ("defendant_name", "property_address"),
    LegalDomain.CHEQUE_BOUNCE: ("drawer_name", "drawer_address"),
    LegalDomain.GENERAL_LEGAL_QUERY: ("parties_involved", None),
}


def _extract_plaintiff(domain: LegalDomain, entities: dict) -> dict | None:
    name_key, addr_key = _PLAINTIFF_KEYS.get(domain, (None, None))
    if not name_key:
        return None
    name = entities.get(name_key)
    if not name:
        return None
    result = {"name": str(name)}
    if addr_key:
        result["address"] = entities.get(addr_key)
    return result


def _extract_defendant(domain: LegalDomain, entities: dict) -> dict | None:
    name_key, addr_key = _DEFENDANT_KEYS.get(domain, (None, None))
    if not name_key:
        return None
    name = entities.get(name_key)
    if not name:
        return None
    result = {"name": str(name)}
    if addr_key:
        result["address"] = entities.get(addr_key)
    return result


def _compose_facts(domain: LegalDomain, entities: dict) -> str:
    """Compose a factual summary from extracted entities."""
    parts = []

    if domain == LegalDomain.RENT_DEPOSIT_DISPUTE:
        parts.append(f"The tenant vacated the premises at {entities.get('property_address', 'the rented property')}.")
        if entities.get("deposit_amount"):
            parts.append(f"A security deposit of INR {entities['deposit_amount']} was paid at the commencement of the tenancy.")
        if entities.get("vacated_date"):
            parts.append(f"The premises were vacated on {entities['vacated_date']}.")
        parts.append("Despite repeated requests, the landlord has failed to return the security deposit.")

    elif domain == LegalDomain.CONSUMER_COMPLAINT:
        parts.append(f"The complainant purchased {entities.get('product_service', 'a product/service')} from {entities.get('opposite_party_name', 'the opposite party')}.")
        if entities.get("amount_paid"):
            parts.append(f"An amount of INR {entities['amount_paid']} was paid.")
        if entities.get("defect_description"):
            parts.append(f"Defect/Deficiency: {entities['defect_description']}.")

    elif domain == LegalDomain.CHEQUE_BOUNCE:
        parts.append(f"A cheque bearing number {entities.get('cheque_number', 'N/A')} dated {entities.get('cheque_date', 'N/A')} for INR {entities.get('cheque_amount', 'N/A')} was issued by the drawer.")
        parts.append(f"The cheque was dishonoured by {entities.get('bank_name', 'the bank')} with the reason: {entities.get('bounce_reason', 'N/A')}.")

    elif domain == LegalDomain.EMPLOYMENT_DISPUTE:
        parts.append(f"The employee was employed as {entities.get('designation', 'N/A')} at {entities.get('employer_name', 'the employer')}.")
        if entities.get("grievance"):
            parts.append(f"Grievance: {entities['grievance']}.")

    elif domain == LegalDomain.PROPERTY_DISPUTE:
        parts.append(f"The dispute concerns the property located at {entities.get('property_address', 'N/A')}.")
        if entities.get("dispute_nature"):
            parts.append(f"Nature of dispute: {entities['dispute_nature']}.")

    else:
        if entities.get("detailed_description"):
            parts.append(entities["detailed_description"])

    return " ".join(parts) if parts else "Facts to be provided."


def _compose_relief(domain: LegalDomain, entities: dict) -> str:
    """Compose the relief sought based on domain and entities."""
    if domain == LegalDomain.RENT_DEPOSIT_DISPUTE:
        amount = entities.get("deposit_amount", "the full deposit")
        return f"Return of security deposit amounting to INR {amount} along with interest."

    if domain == LegalDomain.CONSUMER_COMPLAINT:
        amount = entities.get("amount_paid", "the amount paid")
        return f"Refund of INR {amount}, compensation for deficiency in service, and costs of the complaint."

    if domain == LegalDomain.CHEQUE_BOUNCE:
        amount = entities.get("cheque_amount", "the cheque amount")
        return f"Payment of INR {amount} along with interest and costs, failing which criminal proceedings under Section 138 NI Act."

    if domain == LegalDomain.EMPLOYMENT_DISPUTE:
        return "Appropriate relief including compensation, reinstatement, or settlement of dues as applicable."

    if domain == LegalDomain.PROPERTY_DISPUTE:
        return "Declaration of rightful ownership, injunction against encroachment, and damages as applicable."

    return "Appropriate legal remedy as per the facts and applicable law."
