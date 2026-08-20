"""
ChromaDB Seed Script — Indian Legal Corpus

Seeds the ChromaDB vector store with sample Indian legal documents.
These provide the RAG pipeline with foundational legal knowledge.

Usage:
    cd backend
    python scripts/seed_chroma.py

You can add more documents by extending the LEGAL_DOCUMENTS list below.
Each document needs: id, text, act_name, section, domain.
"""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.chromadb_store import initialize, LegalCorpusStore


# ── Sample Indian Legal Corpus ───────────────────────────────────
# Add your documents here. Each entry is a chunk of legal text with metadata.

LEGAL_DOCUMENTS = [
    # ── Model Tenancy Act ────────────────────────────────────
    {
        "id": "mta_s4_1",
        "text": (
            "Section 4(1) of the Model Tenancy Act, 2021: Every landlord shall enter into "
            "a written tenancy agreement with the tenant for each tenancy. The agreement shall "
            "be executed between the landlord and the tenant within two months from the date of "
            "the commencement of the tenancy."
        ),
        "act_name": "Model Tenancy Act, 2021",
        "section": "Section 4(1)",
        "domain": "rent_deposit_dispute",
    },
    {
        "id": "mta_s12_1",
        "text": (
            "Section 12 of the Model Tenancy Act, 2021: The landlord shall refund the security "
            "deposit to the tenant at the time of taking back the possession of the premises, "
            "after deducting any arrears of rent, cost of repair for damage to premises (beyond "
            "normal wear and tear), and any other dues payable by the tenant. The security deposit "
            "shall not exceed two months' rent for residential premises."
        ),
        "act_name": "Model Tenancy Act, 2021",
        "section": "Section 12",
        "domain": "rent_deposit_dispute",
    },
    {
        "id": "mta_s14",
        "text": (
            "Section 14 of the Model Tenancy Act, 2021: If a landlord fails to refund the "
            "security deposit as per Section 12, the tenant may file a complaint with the Rent "
            "Authority. The Rent Authority may direct the landlord to return the security deposit "
            "with interest at a rate prescribed by the Government."
        ),
        "act_name": "Model Tenancy Act, 2021",
        "section": "Section 14",
        "domain": "rent_deposit_dispute",
    },
    {
        "id": "mta_s21",
        "text": (
            "Section 21 of the Model Tenancy Act, 2021: A landlord may terminate a tenancy by "
            "giving a notice to the tenant of not less than two months if the tenant has failed "
            "to pay rent for two consecutive months."
        ),
        "act_name": "Model Tenancy Act, 2021",
        "section": "Section 21",
        "domain": "rent_deposit_dispute",
    },

    # ── Consumer Protection Act, 2019 ───────────────────────
    {
        "id": "cpa_s2_7",
        "text": (
            "Section 2(7) of the Consumer Protection Act, 2019: 'Consumer' means any person who "
            "buys any goods for a consideration which has been paid or promised or partly paid and "
            "partly promised, or under any system of deferred payment, and includes any user of such "
            "goods when such use is made with the approval of the buyer."
        ),
        "act_name": "Consumer Protection Act, 2019",
        "section": "Section 2(7)",
        "domain": "consumer_complaint",
    },
    {
        "id": "cpa_s2_11",
        "text": (
            "Section 2(11) of the Consumer Protection Act, 2019: 'Deficiency' means any fault, "
            "imperfection, shortcoming or inadequacy in the quality, nature and manner of performance "
            "which is required to be maintained by or under any law for the time being in force or "
            "has been undertaken to be performed by a person in pursuance of a contract."
        ),
        "act_name": "Consumer Protection Act, 2019",
        "section": "Section 2(11)",
        "domain": "consumer_complaint",
    },
    {
        "id": "cpa_s35",
        "text": (
            "Section 35 of the Consumer Protection Act, 2019: A consumer complaint shall be filed "
            "in the District Commission if the value of goods or services paid as consideration does "
            "not exceed one crore rupees. For values between one crore and ten crore rupees, the "
            "complaint shall be filed in the State Commission."
        ),
        "act_name": "Consumer Protection Act, 2019",
        "section": "Section 35",
        "domain": "consumer_complaint",
    },
    {
        "id": "cpa_s39",
        "text": (
            "Section 39 of the Consumer Protection Act, 2019: The District Commission shall, on "
            "admission of a complaint, refer the same to mediation within five days where the "
            "Commission considers it appropriate."
        ),
        "act_name": "Consumer Protection Act, 2019",
        "section": "Section 39",
        "domain": "consumer_complaint",
    },

    # ── Negotiable Instruments Act (Cheque Bounce) ───────────
    {
        "id": "nia_s138",
        "text": (
            "Section 138 of the Negotiable Instruments Act, 1881: Where any cheque drawn by a "
            "person on an account maintained by him with a banker for payment of any amount of "
            "money to another person from out of that account for the discharge, in whole or in "
            "part, of any debt or other liability, is returned by the bank unpaid, such person "
            "shall be deemed to have committed an offence and shall be punishable with imprisonment "
            "for a term which may extend to two years, or with fine which may extend to twice the "
            "amount of the cheque, or with both."
        ),
        "act_name": "Negotiable Instruments Act, 1881",
        "section": "Section 138",
        "domain": "cheque_bounce",
    },
    {
        "id": "nia_s142",
        "text": (
            "Section 142 of the Negotiable Instruments Act, 1881: A complaint under Section 138 "
            "can be filed only by the payee or the holder in due course. The complaint must be "
            "filed within one month from the date on which the cause of action arises."
        ),
        "act_name": "Negotiable Instruments Act, 1881",
        "section": "Section 142",
        "domain": "cheque_bounce",
    },
    {
        "id": "nia_s141",
        "text": (
            "Section 141 of the Negotiable Instruments Act, 1881: If the person committing an "
            "offence under Section 138 is a company, every person who, at the time the offence "
            "was committed, was in charge of and was responsible for the conduct of the business "
            "of the company shall be deemed to be guilty of the offence."
        ),
        "act_name": "Negotiable Instruments Act, 1881",
        "section": "Section 141",
        "domain": "cheque_bounce",
    },

    # ── Indian Contract Act ──────────────────────────────────
    {
        "id": "ica_s73",
        "text": (
            "Section 73 of the Indian Contract Act, 1872: When a contract has been broken, the "
            "party who suffers by such breach is entitled to receive, as compensation for any loss "
            "or damage caused to him thereby, such compensation as naturally arose in the usual "
            "course of things from such breach."
        ),
        "act_name": "Indian Contract Act, 1872",
        "section": "Section 73",
        "domain": "general_legal_query",
    },
    {
        "id": "ica_s27",
        "text": (
            "Section 27 of the Indian Contract Act, 1872: Every agreement by which anyone is "
            "restrained from exercising a lawful profession, trade or business of any kind, is "
            "to that extent void. This provision is relevant in employment disputes involving "
            "non-compete clauses."
        ),
        "act_name": "Indian Contract Act, 1872",
        "section": "Section 27",
        "domain": "employment_dispute",
    },

    # ── Payment of Gratuity Act ──────────────────────────────
    {
        "id": "pga_s4",
        "text": (
            "Section 4 of the Payment of Gratuity Act, 1972: Gratuity shall be payable to an "
            "employee on the termination of his employment after he has rendered continuous "
            "service for not less than five years. The employer shall pay gratuity at the rate "
            "of fifteen days' wages for every completed year of service."
        ),
        "act_name": "Payment of Gratuity Act, 1972",
        "section": "Section 4",
        "domain": "employment_dispute",
    },

    # ── Transfer of Property Act ─────────────────────────────
    {
        "id": "tpa_s54",
        "text": (
            "Section 54 of the Transfer of Property Act, 1882: 'Sale' is a transfer of ownership "
            "in exchange for a price paid or promised or part-paid and part-promised. A contract "
            "for the sale of immovable property of the value of one hundred rupees or upwards shall "
            "be made by a registered instrument."
        ),
        "act_name": "Transfer of Property Act, 1882",
        "section": "Section 54",
        "domain": "property_dispute",
    },
    {
        "id": "tpa_s52",
        "text": (
            "Section 52 of the Transfer of Property Act, 1882 (Doctrine of Lis Pendens): During "
            "the pendency of any suit or proceeding respecting a right to immovable property, the "
            "property cannot be transferred or otherwise dealt with by any party to the suit so as "
            "to affect the rights of any other party."
        ),
        "act_name": "Transfer of Property Act, 1882",
        "section": "Section 52",
        "domain": "property_dispute",
    },
]


def seed():
    """Seed ChromaDB with the sample legal corpus."""
    print("Initializing ChromaDB...")
    initialize()

    store = LegalCorpusStore

    # Check if already seeded
    stats = store.get_stats()
    if stats["document_count"] > 0:
        print(f"Collection already has {stats['document_count']} documents.")
        response = input("Re-seed? This will add duplicates. (y/N): ").strip().lower()
        if response != "y":
            print("Skipping seed. Done.")
            return

    documents = [d["text"] for d in LEGAL_DOCUMENTS]
    metadatas = [
        {
            "act_name": d["act_name"],
            "section": d["section"],
            "domain": d["domain"],
        }
        for d in LEGAL_DOCUMENTS
    ]
    ids = [d["id"] for d in LEGAL_DOCUMENTS]

    print(f"Seeding {len(documents)} legal documents...")
    store.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    stats = store.get_stats()
    print(f"Done! Collection now has {stats['document_count']} documents.")


if __name__ == "__main__":
    seed()
