"""
Legal domain definitions and per-domain entity schemas.

Each supported case type has a corresponding set of fields the Entity Extractor
must produce. These schemas drive both the LLM prompt and the form the frontend
renders for the user.
"""

from enum import Enum
from dataclasses import dataclass, field


class LegalDomain(str, Enum):
    """Supported legal case classifications."""

    RENT_DEPOSIT_DISPUTE = "rent_deposit_dispute"
    CONSUMER_COMPLAINT = "consumer_complaint"
    EMPLOYMENT_DISPUTE = "employment_dispute"
    PROPERTY_DISPUTE = "property_dispute"
    CHEQUE_BOUNCE = "cheque_bounce"
    GENERAL_LEGAL_QUERY = "general_legal_query"


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Output of the Intent Router step."""

    domain: LegalDomain
    confidence: float
    summary: str


@dataclass(frozen=True, slots=True)
class RAGResult:
    """A single retrieved legal clause from ChromaDB or a live government web search."""

    text: str
    act_name: str
    section: str
    relevance_score: float


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Specification for a single form field."""

    key: str
    label: str
    field_type: str = "text"  # text, number, date, textarea
    required: bool = True


# ── Per-Domain Entity Schemas ────────────────────────────────────
# These define what the Entity Extractor must produce for each case type,
# and what the frontend renders as a form.

DOMAIN_FIELD_SCHEMAS: dict[LegalDomain, list[FieldSpec]] = {
    LegalDomain.RENT_DEPOSIT_DISPUTE: [
        FieldSpec(key="landlord_name", label="Landlord's Full Name"),
        FieldSpec(key="tenant_name", label="Tenant's Full Name"),
        FieldSpec(key="property_address", label="Rented Property Address", field_type="textarea"),
        FieldSpec(key="deposit_amount", label="Deposit Amount (INR)", field_type="number"),
        FieldSpec(key="rent_amount", label="Monthly Rent (INR)", field_type="number"),
        FieldSpec(key="lease_start_date", label="Lease Start Date", field_type="date"),
        FieldSpec(key="vacated_date", label="Date Premises Vacated", field_type="date"),
    ],
    LegalDomain.CONSUMER_COMPLAINT: [
        FieldSpec(key="complainant_name", label="Complainant's Full Name"),
        FieldSpec(key="opposite_party_name", label="Company / Seller Name"),
        FieldSpec(key="opposite_party_address", label="Company / Seller Address", field_type="textarea"),
        FieldSpec(key="product_service", label="Product or Service"),
        FieldSpec(key="purchase_date", label="Date of Purchase", field_type="date"),
        FieldSpec(key="amount_paid", label="Amount Paid (INR)", field_type="number"),
        FieldSpec(key="defect_description", label="Description of Defect / Deficiency", field_type="textarea"),
    ],
    LegalDomain.EMPLOYMENT_DISPUTE: [
        FieldSpec(key="employee_name", label="Employee's Full Name"),
        FieldSpec(key="employer_name", label="Employer / Company Name"),
        FieldSpec(key="employer_address", label="Employer Address", field_type="textarea"),
        FieldSpec(key="designation", label="Designation / Role"),
        FieldSpec(key="joining_date", label="Date of Joining", field_type="date"),
        FieldSpec(key="termination_date", label="Date of Termination", field_type="date", required=False),
        FieldSpec(key="salary", label="Monthly Salary (INR)", field_type="number"),
        FieldSpec(key="grievance", label="Nature of Grievance", field_type="textarea"),
    ],
    LegalDomain.PROPERTY_DISPUTE: [
        FieldSpec(key="plaintiff_name", label="Your Full Name"),
        FieldSpec(key="defendant_name", label="Opposing Party's Full Name"),
        FieldSpec(key="property_address", label="Disputed Property Address", field_type="textarea"),
        FieldSpec(key="property_type", label="Type of Property (Residential / Commercial / Land)"),
        FieldSpec(key="dispute_nature", label="Nature of Dispute", field_type="textarea"),
        FieldSpec(key="property_value", label="Approximate Property Value (INR)", field_type="number", required=False),
    ],
    LegalDomain.CHEQUE_BOUNCE: [
        FieldSpec(key="payee_name", label="Payee's Full Name (You)"),
        FieldSpec(key="drawer_name", label="Drawer's Full Name (Cheque Issuer)"),
        FieldSpec(key="drawer_address", label="Drawer's Address", field_type="textarea"),
        FieldSpec(key="cheque_number", label="Cheque Number"),
        FieldSpec(key="cheque_amount", label="Cheque Amount (INR)", field_type="number"),
        FieldSpec(key="cheque_date", label="Date on Cheque", field_type="date"),
        FieldSpec(key="bank_name", label="Drawer's Bank Name"),
        FieldSpec(key="bounce_reason", label="Reason for Dishonour"),
        FieldSpec(key="bounce_date", label="Date of Dishonour", field_type="date"),
    ],
    LegalDomain.GENERAL_LEGAL_QUERY: [
        FieldSpec(key="query_subject", label="Subject of Your Query"),
        FieldSpec(key="detailed_description", label="Detailed Description", field_type="textarea"),
        FieldSpec(key="parties_involved", label="Parties Involved", required=False),
    ],
}
