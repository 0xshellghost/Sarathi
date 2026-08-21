# Sarathi AI Models

Training, fine-tuning, and serving pipelines for the three AI models that power the Sarathi backend.

## Models Overview

```
ai_models/
├── chat_llm/              # P0 — Powers intent, explanation, and extraction
│   ├── Modelfile           # Ollama Modelfile (use immediately)
│   ├── prepare_data.py     # Convert training data for fine-tuning
│   ├── train.py            # LoRA fine-tuning with Unsloth
│   ├── evaluate.py         # Benchmark all 3 task types
│   └── training_data/
│       └── legal_conversations.jsonl  # 25 training conversations
│
├── embedding/              # P1 — Powers ChromaDB vector search
│   ├── Modelfile           # Ollama Modelfile
│   ├── train.py            # Fine-tune sentence-transformer
│   ├── serve.py            # Standalone embedding server
│   └── training_data/
│       └── legal_pairs.jsonl  # 16 triplet training pairs
│
├── stt/                    # P2 — Powers voice search
│   ├── train.py            # Fine-tune Whisper
│   └── serve.py            # FastAPI endpoint for transcription
│
├── requirements.txt        # ML dependencies
└── README.md               # This file
```

---

## Quick Start (No Fine-Tuning)

You can get the entire system running in minutes using the pre-built Ollama Modelfiles with base models:

```bash
# 1. Install Ollama (if not already)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull base models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 3. Create Sarathi models from Modelfiles
cd ai_models/chat_llm
ollama create sarathi-legal -f Modelfile

cd ../embedding
ollama create sarathi-embed -f Modelfile

# 4. Test
ollama run sarathi-legal "My landlord won't return my deposit of 50000"
```

---

## Fine-Tuning Guide

### Prerequisites

```bash
# Create a separate virtual environment for ML work
python3 -m venv ml_venv
source ml_venv/bin/activate
pip install -r requirements.txt
```

### P0: Chat LLM

The chat model handles three tasks: intent classification, legal explanations, and entity extraction. Fine-tuning improves accuracy on Indian legal domain.

```bash
cd chat_llm

# Step 1: Prepare training data
python prepare_data.py

# Step 2: Fine-tune (requires GPU with 8GB+ VRAM)
python train.py

# Step 3: The script auto-exports to GGUF. Load in Ollama:
# Update Modelfile: FROM ./output/sarathi-legal-lora/gguf/unsloth.Q4_K_M.gguf
ollama create sarathi-legal -f Modelfile

# Step 4: Evaluate
python evaluate.py
```

**Adding more training data:** Append conversations to `training_data/legal_conversations.jsonl` in the same JSONL format. Each line needs `messages` with `system`, `user`, and `assistant` roles.

### P1: Embedding Model

The embedding model converts legal text into vectors for ChromaDB similarity search.

```bash
cd embedding

# Fine-tune on legal triplets
python train.py

# Evaluate similarity scores
python train.py --eval-only

# Option A: Serve via Ollama (update Modelfile FROM path)
ollama create sarathi-embed -f Modelfile

# Option B: Serve via standalone FastAPI server
python serve.py --model ./output/sarathi-embed --port 9001
```

### P2: STT Model

The speech-to-text model transcribes Hindi/English audio for voice search.

```bash
cd stt

# You need audio training data first:
# Create training_data/metadata.csv and training_data/audio/*.wav
# See train.py for detailed instructions and data sources.

# Fine-tune
python train.py

# Serve
python serve.py --model ./output/sarathi-stt --port 9002

# Or use base Whisper without fine-tuning:
python serve.py --model openai/whisper-small --port 9002

# Then set in backend/.env:
# STT_ENDPOINT=http://localhost:9002/transcribe
```

---

## Training Data

### Chat LLM Data Format

Each line in `legal_conversations.jsonl` is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "Classify the user's legal problem..."},
    {"role": "user", "content": "My landlord won't return my deposit"},
    {"role": "assistant", "content": "{\"domain\": \"rent_deposit_dispute\", ...}"}
  ]
}
```

The current dataset has **25 conversations** covering:
- 14 intent classification examples (all 6 domains)
- 5 legal explanation examples (with Hindi-English mix)
- 6 entity extraction examples (all domain schemas)

**To improve model quality:** Add at least 200-500 conversations per task type.

### Embedding Data Format

Each line in `legal_pairs.jsonl` is a triplet:

```json
{
  "query": "user's legal question",
  "positive": "relevant statute/section text",
  "negative": "irrelevant statute/section text"
}
```

The current dataset has **16 triplets**. For production quality, aim for 500+ triplets.

### STT Audio Data Sources

For Hindi-English audio data, use:
- [Mozilla Common Voice (Hindi)](https://commonvoice.mozilla.org/hi/datasets)
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs)
- [AI4Bharat IndicVoices](https://ai4bharat.iitm.ac.in/indicvoices/)
- Record your own legal domain audio

---

## Backend Integration

After training and deploying models, update the backend's `.env`:

```bash
# Chat LLM (via Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=sarathi-legal

# Embedding Model
# Option A: Via Ollama
OLLAMA_EMBED_MODEL=sarathi-embed

# Option B: Via standalone server
# Modify backend/db/chromadb_store.py to call http://localhost:9001/api/embeddings

# STT Model
STT_ENDPOINT=http://localhost:9002/transcribe
```

---

## Hardware Recommendations

| Model | Minimum GPU | Recommended | CPU Feasible? |
|-------|-------------|-------------|---------------|
| Chat LLM (8B, 4-bit LoRA) | 8GB VRAM | 16GB VRAM | No |
| Embedding (MiniLM) | 4GB VRAM | 8GB VRAM | Yes (~30 min) |
| STT (Whisper-small) | 8GB VRAM | 16GB VRAM | Yes (slow) |

For inference (serving), requirements are lower than training.
