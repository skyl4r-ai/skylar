# synth-gen — Synthetic Italian Conversation Generator

Agentic pipeline that reads pre-training documents (`<bos>...<eos>` format) and generates high-quality Italian conversation turns in JSONL (ChatML) format for fine-tuning.

## Architecture

```
Dataset Dir → Document Parser → [Analyst Agent] → [Turn Builder Agent] → [Validator Agent] → Dedup → JSONL
```

Three LLM agents collaborate per document:
1. **Analyst** — classifies doc type, extracts topics, plans turn distribution
2. **Turn Builder** — generates system prompt + conversation per type (short_qa, long_qa, expert_deep, negative, json_output, tool_call)
3. **Validator** — quality gate scoring 1-10, rejects < 7

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start vLLM server first
vllm serve <model> --port 8000

# Basic usage
python synth_gen.py generate --dataset ./pretrain_data --messages 500

# Full options
python synth_gen.py generate \
  --dataset ./pretrain_data \
  --messages 1000 \
  --output ./output/dataset.jsonl \
  --seed 42

# Skip validation for speed
python synth_gen.py generate -d ./pretrain_data -m 200 --skip-validation

# Check stats
python synth_gen.py stats --output ./output/dataset.jsonl
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_API_KEY` | `not-needed` | API key (if required) |
| `VLLM_MODEL` | (auto-detect) | Model name to use |
| `TEMP_ANALYSIS` | `0.15` | Temperature for analyst agent |
| `TEMP_GENERATION` | `0.55` | Temperature for turn builder |
| `TEMP_VALIDATION` | `0.10` | Temperature for validator |
| `MAX_RETRIES_PER_TURN` | `3` | Retries per failed turn |
| `MAX_DOC_CHARS` | `16384` | Max chars per document sent to LLM |

## Crash Safety

- JSONL output is append-only with `fsync` on every write
- Journal file (`.journal.json`) tracks processed docs and generated turn hashes
- Resume from interruption: just re-run the same command
- MD5 deduplication prevents duplicate conversations

## Output Format

Each line in the JSONL:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Tool call turns include `<tool_call>` / `<tool_response>` tags inline in message content, following ChatML conventions.

## Dataset Format

Input files should contain documents wrapped in `<bos>...<eos>` tags. JSON documents use `<bos>JSON...<eos>`. Files without markers are treated as single documents (auto-chunked if large).
