"""
Evaluation Script for Sarathi Legal Chat LLM

Tests the fine-tuned model across three task types:
  1. Intent Classification — measures domain accuracy and confidence calibration
  2. Entity Extraction — measures field extraction precision
  3. Legal Explanation — qualitative scoring (manual review + basic heuristics)

Usage:
    # Evaluate against the live Ollama model
    python evaluate.py

    # Evaluate a local model checkpoint
    python evaluate.py --local-model ./output/sarathi-legal-lora

    # Save results to file
    python evaluate.py --output results.json
"""

import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sarathi.eval")

SCRIPT_DIR = Path(__file__).parent


# ── Evaluation Test Cases ────────────────────────────────────────

INTENT_TESTS = [
    {
        "input": "My builder has delayed possession of my flat by 3 years despite RERA registration",
        "expected_domain": "property_dispute",
    },
    {
        "input": "Zomato delivered wrong food order and refused refund of 850 rupees",
        "expected_domain": "consumer_complaint",
    },
    {
        "input": "Mera landlord security deposit nahi de raha, 6 mahine ho gaye",
        "expected_domain": "rent_deposit_dispute",
    },
    {
        "input": "Company ne 10 saal baad bhi gratuity nahi di",
        "expected_domain": "employment_dispute",
    },
    {
        "input": "Received a cheque of 1.5 lakhs that bounced due to account closure",
        "expected_domain": "cheque_bounce",
    },
    {
        "input": "How do I file an RTI application?",
        "expected_domain": "general_legal_query",
    },
    {
        "input": "Tenant has sub-let my property without permission",
        "expected_domain": "rent_deposit_dispute",
    },
    {
        "input": "Online tuition class took advance fee of 20000 and shut down without delivering any classes",
        "expected_domain": "consumer_complaint",
    },
]

ENTITY_TESTS = [
    {
        "input": "I'm Deepa Nair. Landlord Mr. Suresh Menon won't return my deposit of 60000. Property at 22 MG Road, Kochi. I vacated on 1st April 2024. Rent was 18000.",
        "domain": "rent_deposit_dispute",
        "expected": {
            "landlord_name": "Suresh Menon",
            "tenant_name": "Deepa Nair",
            "deposit_amount": 60000,
            "rent_amount": 18000,
        },
    },
    {
        "input": "I am Karan Singh. Cheque 789012 for 5 lakhs from Ajay Mehta of 9 Park Street Kolkata bounced on 20 June 2024 from SBI due to insufficient funds. Cheque dated 15 June 2024.",
        "domain": "cheque_bounce",
        "expected": {
            "drawer_name": "Ajay Mehta",
            "payee_name": "Karan Singh",
            "cheque_amount": 500000,
            "bank_name": "SBI",
        },
    },
]

EXPLANATION_TESTS = [
    {
        "input": "My employer hasn't paid my salary for 3 months",
        "must_mention": ["Payment of Wages Act", "labour commissioner", "complaint"],
    },
    {
        "input": "Landlord is not returning deposit after I vacated 4 months ago",
        "must_mention": ["Model Tenancy Act", "Section 12", "notice", "Rent Authority"],
    },
]


@dataclass
class EvalResults:
    """Accumulates evaluation metrics."""
    intent_correct: int = 0
    intent_total: int = 0
    entity_field_correct: int = 0
    entity_field_total: int = 0
    explanation_keywords_found: int = 0
    explanation_keywords_total: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct / max(self.intent_total, 1)

    @property
    def entity_precision(self) -> float:
        return self.entity_field_correct / max(self.entity_field_total, 1)

    @property
    def explanation_coverage(self) -> float:
        return self.explanation_keywords_found / max(self.explanation_keywords_total, 1)

    def summary(self) -> dict:
        return {
            "intent_accuracy": f"{self.intent_accuracy:.1%}",
            "entity_field_precision": f"{self.entity_precision:.1%}",
            "explanation_keyword_coverage": f"{self.explanation_coverage:.1%}",
            "errors": self.errors,
        }


async def call_ollama(
    messages: list[dict],
    json_mode: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = "sarathi-legal",
) -> str:
    """Make a blocking call to the Ollama API."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


async def eval_intent_classification(results: EvalResults, **kwargs) -> None:
    """Evaluate intent classification accuracy."""
    logger.info("\n📋 Evaluating Intent Classification (%d tests)", len(INTENT_TESTS))

    for test in INTENT_TESTS:
        results.intent_total += 1

        messages = [
            {"role": "system", "content": 'Classify the user\'s legal problem into a domain. Respond with ONLY a JSON object: {"domain": "...", "confidence": 0.0-1.0, "summary": "..."}'},
            {"role": "user", "content": test["input"]},
        ]

        try:
            raw = await call_ollama(messages, json_mode=True, **kwargs)
            data = json.loads(raw)
            predicted = data.get("domain", "")

            if predicted == test["expected_domain"]:
                results.intent_correct += 1
                logger.info("  ✅ '%s...' → %s", test["input"][:50], predicted)
            else:
                logger.info("  ❌ '%s...' → %s (expected %s)",
                           test["input"][:50], predicted, test["expected_domain"])

        except Exception as exc:
            results.errors.append(f"Intent: {exc}")
            logger.error("  ⚠ Error on '%s...': %s", test["input"][:40], exc)


async def eval_entity_extraction(results: EvalResults, **kwargs) -> None:
    """Evaluate entity extraction precision."""
    logger.info("\n🔍 Evaluating Entity Extraction (%d tests)", len(ENTITY_TESTS))

    for test in ENTITY_TESTS:
        field_list = ", ".join(test["expected"].keys())
        messages = [
            {"role": "system", "content": f"Extract entities from the user's description. Respond with ONLY a valid JSON object. Fields: {field_list}"},
            {"role": "user", "content": test["input"]},
        ]

        try:
            raw = await call_ollama(messages, json_mode=True, **kwargs)
            data = json.loads(raw)

            for key, expected_val in test["expected"].items():
                results.entity_field_total += 1
                actual = data.get(key)

                # Flexible matching: string containment or numeric equality
                match = False
                if isinstance(expected_val, str) and isinstance(actual, str):
                    match = expected_val.lower() in actual.lower() or actual.lower() in expected_val.lower()
                elif isinstance(expected_val, (int, float)):
                    match = actual == expected_val
                else:
                    match = str(actual) == str(expected_val)

                if match:
                    results.entity_field_correct += 1
                    logger.info("  ✅ %s = %s", key, actual)
                else:
                    logger.info("  ❌ %s = %s (expected %s)", key, actual, expected_val)

        except Exception as exc:
            results.errors.append(f"Entity: {exc}")
            logger.error("  ⚠ Error: %s", exc)


async def eval_explanations(results: EvalResults, **kwargs) -> None:
    """Evaluate legal explanation quality via keyword coverage."""
    logger.info("\n📖 Evaluating Legal Explanations (%d tests)", len(EXPLANATION_TESTS))

    for test in EXPLANATION_TESTS:
        messages = [
            {"role": "system", "content": "Explain the user's legal rights clearly and empathetically. Reference specific Indian acts and sections."},
            {"role": "user", "content": test["input"]},
        ]

        try:
            response = await call_ollama(messages, **kwargs)
            response_lower = response.lower()

            for keyword in test["must_mention"]:
                results.explanation_keywords_total += 1
                if keyword.lower() in response_lower:
                    results.explanation_keywords_found += 1
                    logger.info("  ✅ Found '%s'", keyword)
                else:
                    logger.info("  ❌ Missing '%s'", keyword)

        except Exception as exc:
            results.errors.append(f"Explanation: {exc}")
            logger.error("  ⚠ Error: %s", exc)


async def run_evaluation(
    base_url: str = "http://localhost:11434",
    model: str = "sarathi-legal",
    output_path: Path | None = None,
):
    """Run the complete evaluation suite."""
    kwargs = {"base_url": base_url, "model": model}
    results = EvalResults()

    logger.info("=" * 60)
    logger.info("🧪 Sarathi Legal LLM — Evaluation Suite")
    logger.info("   Model: %s @ %s", model, base_url)
    logger.info("=" * 60)

    await eval_intent_classification(results, **kwargs)
    await eval_entity_extraction(results, **kwargs)
    await eval_explanations(results, **kwargs)

    # Print summary
    summary = results.summary()
    logger.info("\n" + "=" * 60)
    logger.info("📊 Results Summary")
    logger.info("=" * 60)
    logger.info("  Intent Classification Accuracy: %s", summary["intent_accuracy"])
    logger.info("  Entity Extraction Precision:    %s", summary["entity_field_precision"])
    logger.info("  Explanation Keyword Coverage:    %s", summary["explanation_keyword_coverage"])

    if results.errors:
        logger.info("\n  ⚠ %d errors encountered", len(results.errors))

    if output_path:
        output_path.write_text(json.dumps(summary, indent=2))
        logger.info("\n  Results saved to %s", output_path)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Sarathi Legal LLM")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="sarathi-legal")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.base_url, args.model, args.output))


if __name__ == "__main__":
    main()
