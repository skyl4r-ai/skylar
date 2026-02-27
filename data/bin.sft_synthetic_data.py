#!/usr/bin/env python3
# filepath: synth_gen.py
"""
Synthetic Conversation Generator for Fine-Tuning
=================================================
Agentic pipeline that reads pre-training documents and generates
high-quality Italian conversation turns in JSONL (ChatML) format.

Designed for multi-GB dataset files — streaming parser, zero full-file loading.

Architecture:
  Dataset Dir --> Streaming Parser (chunk-by-chunk)
       |             yields ParsedDocument one at a time
       v
  [Agent: Analyst]        -- classifies doc, plans turn distribution
       |
       v
  [Agent: Turn Builder]   -- generates system + conversation per type
       |
       v
  [Agent: Validator]      -- quality gate, accept/reject
       |
       v
  [Dedup via MD5]         -- skip semantic duplicates
       |
       v
  [JSONL Writer]          -- incremental append (crash-safe)
       |
       v
  [Final Shuffle]         -- randomize order at the end
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
import uuid
from collections.abc import Generator, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

# Load .env from script directory (before any os.getenv)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — use shell env only

import structlog
import typer
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Anthropic SDK — optional, only needed for provider=anthropic
try:
    import anthropic as _anthropic_mod

    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore[assignment]
    _HAS_ANTHROPIC = False

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log: structlog.stdlib.BoundLogger = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Provider: "openai" (also works with vLLM), "anthropic", or "" for auto-detect
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")

# OpenAI / vLLM config (backward compatible with VLLM_* vars)
OPENAI_BASE_URL: str = (
    os.getenv("VLLM_BASE_URL")
    or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
)
OPENAI_API_KEY: str = (
    os.getenv("VLLM_API_KEY")
    or os.getenv("OPENAI_API_KEY", "not-needed")
)
OPENAI_MODEL: str = (
    os.getenv("VLLM_MODEL")
    or os.getenv("OPENAI_MODEL", "")
)

# Anthropic config
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

TEMP_ANALYSIS: float = float(os.getenv("TEMP_ANALYSIS", "0.15"))
TEMP_GENERATION: float = float(os.getenv("TEMP_GENERATION", "0.55"))
TEMP_VALIDATION: float = float(os.getenv("TEMP_VALIDATION", "0.10"))

MAX_RETRIES_PER_TURN: int = int(os.getenv("MAX_RETRIES_PER_TURN", "5"))
MAX_DOC_CHARS: int = int(os.getenv("MAX_DOC_CHARS", "12000"))

# Streaming parser — chunk size for file reads
READ_CHUNK_SIZE: int = int(os.getenv("READ_CHUNK_SIZE", str(256 * 1024)))  # 256KB

# Sentence boundary pattern for clean truncation
_SENTENCE_END = re.compile(r"[.!?;]\s", re.MULTILINE)


def _truncate_clean(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """Truncate text at the last sentence boundary within max_chars.

    Avoids cutting mid-word or mid-sentence, which confuses LLMs into
    thinking the document is 'corrupted'.
    """
    if len(text) <= max_chars:
        return text
    # Search for last sentence-ending punctuation before the limit
    chunk = text[:max_chars]
    matches = list(_SENTENCE_END.finditer(chunk))
    if matches:
        # Cut right after the last sentence-ending punctuation
        last = matches[-1]
        return chunk[: last.end()].rstrip()
    # No sentence boundary found — cut at last whitespace
    last_space = chunk.rfind(" ")
    if last_space > max_chars // 2:
        return chunk[:last_space].rstrip()
    return chunk.rstrip()

app = typer.Typer(
    name="synth-gen",
    help="Generate synthetic Italian conversation turns from pre-training documents.",
    add_completion=False,
)


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════


class TurnType(str, Enum):
    SHORT_QA = "short_qa"
    LONG_QA = "long_qa"
    EXPERT_DEEP = "expert_deep"
    NEGATIVE = "negative"
    JSON_OUTPUT = "json_output"
    TOOL_CALL = "tool_call"


class DocumentAnalysis(BaseModel):
    doc_type: str = Field(
        description="Document type: normativa, circolare, wiki, report_bdi, json_data, comunicato, regolamento, altro"
    )
    language: str = Field(description="Primary language of the document")
    topics: list[str] = Field(description="Main topics covered (3-8 items)")
    key_entities: list[str] = Field(description="Key entities mentioned")
    complexity: str = Field(description="low, medium, or high")
    summary: str = Field(description="Brief factual summary in Italian, 2-3 sentences")
    turn_plan: dict[str, int] = Field(
        description=(
            "Number of turns to generate per type. Keys: "
            "short_qa, long_qa, expert_deep, negative, json_output, tool_call"
        )
    )


class ConversationMessage(BaseModel):
    role: str = Field(description="system, user, or assistant")
    content: str = Field(description="Message content")


class GeneratedConversation(BaseModel):
    system_prompt: str = Field(description="System prompt for this conversation")
    messages: list[ConversationMessage] = Field(
        description="Ordered list of user/assistant messages (no system)"
    )


class ToolCallConversation(BaseModel):
    system_prompt: str = Field(
        description="System prompt including tool definitions in <tools> XML"
    )
    messages: list[ConversationMessage] = Field(
        description=(
            "Ordered conversation. Assistant tool calls use <tool_call> tags. "
            "Tool responses use role=user with <tool_response> tags."
        )
    )


class ValidationResult(BaseModel):
    accepted: bool = Field(description="Whether the conversation passes quality checks")
    reason: str = Field(description="Explanation for acceptance or rejection")
    score: int = Field(description="Quality score 1-10")


# ═══════════════════════════════════════════════════════════════
# DOMAIN TOOLS FOR TOOL-CALL TURNS
# ═══════════════════════════════════════════════════════════════

DOMAIN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "cerca_normativa",
        "description": "Cerca una normativa, circolare o regolamento nella base dati di Banca d'Italia",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Parole chiave per la ricerca normativa"},
                "tipo": {"type": "string", "enum": ["circolare", "regolamento", "provvedimento", "comunicazione"], "description": "Tipo di normativa da cercare"},
                "anno": {"type": "integer", "description": "Anno di pubblicazione (opzionale)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_tasso_riferimento",
        "description": "Ottieni il tasso di interesse di riferimento BCE per una data specifica",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Data nel formato YYYY-MM-DD"},
                "tipo_tasso": {"type": "string", "enum": ["refinanziamento", "deposito", "marginale", "euribor"], "description": "Tipo di tasso richiesto"},
            },
            "required": ["data", "tipo_tasso"],
        },
    },
    {
        "name": "calcola_rata_mutuo",
        "description": "Calcola la rata mensile di un mutuo dato capitale, tasso e durata",
        "parameters": {
            "type": "object",
            "properties": {
                "capitale": {"type": "number", "description": "Importo del mutuo in euro"},
                "tasso_annuo": {"type": "number", "description": "Tasso di interesse annuo in percentuale"},
                "durata_mesi": {"type": "integer", "description": "Durata del mutuo in mesi"},
                "tipo": {"type": "string", "enum": ["fisso", "variabile"], "description": "Tipo di tasso"},
            },
            "required": ["capitale", "tasso_annuo", "durata_mesi"],
        },
    },
    {
        "name": "get_dati_bilancio",
        "description": "Recupera dati di bilancio aggregati per il sistema bancario italiano",
        "parameters": {
            "type": "object",
            "properties": {
                "anno": {"type": "integer", "description": "Anno di riferimento"},
                "aggregato": {"type": "string", "enum": ["attivo", "passivo", "patrimonio", "sofferenze", "impieghi", "raccolta"], "description": "Tipo di aggregato"},
                "dettaglio": {"type": "string", "enum": ["nazionale", "regionale", "per_dimensione"], "description": "Livello di dettaglio"},
            },
            "required": ["anno", "aggregato"],
        },
    },
    {
        "name": "verifica_codice_abi",
        "description": "Verifica e ottieni informazioni su un codice ABI di un istituto bancario",
        "parameters": {
            "type": "object",
            "properties": {
                "codice_abi": {"type": "string", "description": "Codice ABI a 5 cifre"},
            },
            "required": ["codice_abi"],
        },
    },
    {
        "name": "get_cambio_valuta",
        "description": "Ottieni il tasso di cambio ufficiale BCE per una coppia di valute",
        "parameters": {
            "type": "object",
            "properties": {
                "valuta_base": {"type": "string", "description": "Codice ISO valuta base (es. EUR)"},
                "valuta_target": {"type": "string", "description": "Codice ISO valuta target (es. USD)"},
                "data": {"type": "string", "description": "Data nel formato YYYY-MM-DD"},
            },
            "required": ["valuta_base", "valuta_target"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# JOURNAL MANAGER — Set-based, O(1) lookups
# ═══════════════════════════════════════════════════════════════


class JournalManager:
    """Manages processing state for crash-safe resume.

    Uses sets internally for O(1) lookup instead of O(n) list scans.
    """

    def __init__(self, output_dir: Path) -> None:
        self._path: Path = output_dir / ".journal.json"
        self._processed_docs: set[str] = set()
        self._turn_hashes: set[str] = set()
        self._total_generated: int = 0
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self._processed_docs = set(data.get("processed_docs", []))
                self._turn_hashes = set(data.get("turn_hashes", []))
                self._total_generated = data.get("total_generated", 0)
                log.info(
                    "journal.loaded",
                    total=self._total_generated,
                    docs=len(self._processed_docs),
                    hashes=len(self._turn_hashes),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("journal.corrupted_resetting")

    def _save(self) -> None:
        payload = {
            "processed_docs": list(self._processed_docs),
            "turn_hashes": list(self._turn_hashes),
            "total_generated": self._total_generated,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".journal.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @property
    def total_generated(self) -> int:
        return self._total_generated

    def is_doc_processed(self, doc_hash: str) -> bool:
        return doc_hash in self._processed_docs

    def is_duplicate(self, turn_hash: str) -> bool:
        return turn_hash in self._turn_hashes

    def register_turn(self, turn_hash: str) -> None:
        """Register a turn hash for dedup. Does NOT save or increment counter."""
        self._turn_hashes.add(turn_hash)

    def increment_generated(self) -> None:
        """Increment the generated counter by 1 and persist. Call once per saved turn."""
        self._total_generated += 1
        self._save()

    def mark_doc_processed(self, doc_hash: str) -> None:
        self._processed_docs.add(doc_hash)
        self._save()


# ═══════════════════════════════════════════════════════════════
# STREAMING DOCUMENT PARSER — Chunk-by-chunk, yields 1 doc at a time
# ═══════════════════════════════════════════════════════════════


class ParsedDocument:
    """A single document extracted from a pre-training file."""

    __slots__ = ("content", "is_json", "source_file", "index", "hash")

    def __init__(
        self, content: str, is_json: bool, source_file: str, index: int
    ) -> None:
        self.content = content.strip()
        self.is_json = is_json
        self.source_file = source_file
        self.index = index
        self.hash = hashlib.md5(self.content.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        tag = "JSON" if self.is_json else "DOC"
        return f"<ParsedDocument {tag} src={self.source_file} idx={self.index} len={len(self.content)}>"


def stream_dataset_directory(
    dataset_dir: Path,
) -> Generator[ParsedDocument, None, None]:
    """Stream-parse all files in the dataset directory.

    Reads files in 256KB chunks, extracts <bos>...<eos> blocks via state machine.
    Yields one ParsedDocument at a time — never loads a full file into memory.
    Peak memory per file: ~MAX_DOC_CHARS * 3 (one block buffer).
    """
    if not dataset_dir.is_dir():
        log.error("dataset.not_a_directory", path=str(dataset_dir))
        raise typer.BadParameter(f"Dataset path is not a directory: {dataset_dir}")

    files = sorted(
        p
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix in (".txt", ".jsonl", ".json", ".md", "")
    )

    if not files:
        log.error("dataset.no_files", path=str(dataset_dir))
        raise typer.BadParameter(f"No supported files found in {dataset_dir}")

    total_yielded = 0
    for file_idx, filepath in enumerate(files, 1):
        file_size = filepath.stat().st_size
        file_size_gb = file_size / (1024 ** 3)
        file_size_mb = file_size / (1024 ** 2)
        size_str = f"{file_size_gb:.2f}GB" if file_size_gb >= 1.0 else f"{file_size_mb:.1f}MB"
        log.info(
            "dataset.streaming_file",
            file=filepath.name,
            progress=f"{file_idx}/{len(files)}",
            size=size_str,
        )

        file_doc_count = 0
        try:
            for doc in _stream_file_bos_eos(filepath):
                total_yielded += 1
                file_doc_count += 1
                if file_doc_count % 500 == 0:
                    log.info(
                        "dataset.stream_progress",
                        file=filepath.name,
                        docs_so_far=file_doc_count,
                    )
                yield doc
        except OSError as exc:
            log.warning("dataset.read_error", file=str(filepath), error=str(exc))
            continue

        log.info(
            "dataset.file_done",
            file=filepath.name,
            docs_in_file=file_doc_count,
            total_yielded=total_yielded,
        )

    log.info("dataset.stream_complete", total_docs=total_yielded)


def _stream_file_bos_eos(
    filepath: Path,
) -> Generator[ParsedDocument, None, None]:
    """Stream a single file extracting <bos>...<eos> blocks.

    State machine reads READ_CHUNK_SIZE bytes at a time.
    Only buffers the current block (capped at MAX_DOC_CHARS*3).
    Handles tags split across chunk boundaries via carry-over.
    """
    bos_tag = "<bos>"
    eos_tag = "<eos>"
    tag_max_len = max(len(bos_tag), len(eos_tag))
    max_block_chars = MAX_DOC_CHARS * 3  # safety cap

    doc_index = 0
    inside_block = False
    block_parts: list[str] = []
    block_len = 0
    carry = ""

    with open(filepath, encoding="utf-8", errors="replace") as f:
        while True:
            raw = f.read(READ_CHUNK_SIZE)
            if not raw and not carry:
                break

            data = carry + raw
            carry = ""

            # EOF: data = leftover carry, no new bytes from file.
            # If inside a block, the carry chars were excluded from block_parts
            # in the previous iteration to avoid double-counting. We must add
            # them now before flushing.
            if not raw:
                if inside_block:
                    if data:
                        block_parts.append(data)
                    if block_parts:
                        doc = _make_doc("".join(block_parts), filepath.name, doc_index)
                        if doc:
                            yield doc
                            doc_index += 1
                break

            pos = 0
            data_len = len(data)

            while pos < data_len:
                if not inside_block:
                    bos_pos = data.find(bos_tag, pos)
                    if bos_pos == -1:
                        # Keep tail as carry in case <bos> is split across chunks
                        carry = data[max(0, data_len - tag_max_len + 1):]
                        break
                    inside_block = True
                    block_parts = []
                    block_len = 0
                    pos = bos_pos + len(bos_tag)
                else:
                    eos_pos = data.find(eos_tag, pos)
                    if eos_pos == -1:
                        # Buffer up to (but not including) the carry tail,
                        # so carry chars aren't double-counted.
                        carry_start = max(pos, data_len - tag_max_len + 1)
                        fragment = data[pos:carry_start]
                        if fragment:
                            remaining_cap = max_block_chars - block_len
                            if remaining_cap > 0:
                                to_add = fragment[:remaining_cap]
                                block_parts.append(to_add)
                                block_len += len(to_add)
                        carry = data[carry_start:]
                        break
                    # Found <eos>
                    block_parts.append(data[pos:eos_pos])
                    content = "".join(block_parts)
                    doc = _make_doc(content, filepath.name, doc_index)
                    if doc:
                        yield doc
                        doc_index += 1
                    inside_block = False
                    block_parts = []
                    block_len = 0
                    pos = eos_pos + len(eos_tag)


def _make_doc(
    content: str, source_file: str, index: int
) -> ParsedDocument | None:
    """Create a ParsedDocument from raw block content, or None if too short."""
    content = content.strip()
    if len(content) < 50:
        return None

    is_json = content.upper().startswith("JSON")
    if is_json:
        content = content[4:].strip()
        if len(content) < 20:
            return None

    # Truncate — the LLM only sees MAX_DOC_CHARS anyway
    if len(content) > MAX_DOC_CHARS:
        content = content[:MAX_DOC_CHARS]

    return ParsedDocument(
        content=content,
        is_json=is_json,
        source_file=source_file,
        index=index,
    )


# ═══════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════


class LLMClient:
    """Unified LLM client supporting OpenAI (+ vLLM) and Anthropic.

    Provider detection:
      - Explicit via ``provider`` param or LLM_PROVIDER env var
      - Auto: if ANTHROPIC_API_KEY is set → anthropic, else → openai

    OpenAI path keeps full vLLM compatibility (probe, fallback chain).
    Anthropic path uses native ``messages.parse()`` + prompt caching.
    """

    def __init__(
        self,
        provider: str = "",
        base_url: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        # Resolve provider
        self._provider = (provider or LLM_PROVIDER).lower()
        if not self._provider:
            self._provider = "anthropic" if ANTHROPIC_API_KEY else "openai"

        if self._provider == "anthropic":
            self._init_anthropic(api_key or ANTHROPIC_API_KEY, model or ANTHROPIC_MODEL)
        else:
            self._init_openai(
                base_url or OPENAI_BASE_URL,
                api_key or OPENAI_API_KEY,
                model or OPENAI_MODEL,
            )

    # ── OpenAI / vLLM init ──────────────────────────────────────

    def _init_openai(self, base_url: str, api_key: str, model: str) -> None:
        self._openai = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model or self._detect_model_openai()
        self._mode = self._probe_response_format()
        log.info(
            "llm.initialized",
            provider="openai",
            model=self._model,
            base_url=base_url,
            response_format=self._mode,
        )

    def _detect_model_openai(self) -> str:
        try:
            models = self._openai.models.list()
            if models.data:
                model_id = models.data[0].id
                log.info("llm.auto_detected_model", model=model_id)
                return model_id
        except Exception as exc:
            log.error("llm.model_detection_failed", error=str(exc))
        raise RuntimeError(
            "Cannot detect model. Set OPENAI_MODEL or VLLM_MODEL env var."
        )

    def _probe_response_format(self) -> str:
        """Probe which response_format the server supports."""
        test_messages = [
            {"role": "user", "content": "Reply with: {\"status\": \"ok\"}"},
        ]
        # Try json_schema first
        try:
            self._openai.chat.completions.create(
                model=self._model,
                messages=test_messages,
                max_tokens=20,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Probe",
                        "schema": {
                            "type": "object",
                            "properties": {"status": {"type": "string"}},
                            "required": ["status"],
                        },
                        "strict": True,
                    },
                },
            )
            log.info("llm.probe", result="json_schema supported")
            return "json_schema"
        except Exception as exc:
            log.debug("llm.probe", result=f"json_schema failed: {exc}")
        # Try json_object
        try:
            self._openai.chat.completions.create(
                model=self._model,
                messages=test_messages,
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            log.info("llm.probe", result="json_object supported")
            return "json_object"
        except Exception as exc:
            log.debug("llm.probe", result=f"json_object failed: {exc}")
        log.warning("llm.probe", result="no structured output — plain text fallback")
        return "text"

    # ── Anthropic init ──────────────────────────────────────────

    def _init_anthropic(self, api_key: str, model: str) -> None:
        if not _HAS_ANTHROPIC:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            )
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or add to .env."
            )
        self._anthropic = _anthropic_mod.Anthropic(api_key=api_key)
        self._model = model
        log.info(
            "llm.initialized",
            provider="anthropic",
            model=self._model,
        )

    # ── Unified public API ──────────────────────────────────────

    def generate_structured(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> BaseModel | None:
        """Generate a structured response parsed into a Pydantic model."""
        if self._provider == "anthropic":
            return self._generate_anthropic(
                system, user, response_model, temperature, max_tokens
            )
        return self._generate_openai(
            system, user, response_model, temperature, max_tokens
        )

    # ── OpenAI / vLLM structured generation ─────────────────────

    def _call_llm_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Low-level OpenAI call with retry on transient errors."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                completion = self._openai.chat.completions.create(**kwargs)
                return (completion.choices[0].message.content or "").strip()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = exc
                error_str = str(exc)
                if "400" in error_str and "not supported" in error_str.lower():
                    raise
                if "401" in error_str or "403" in error_str:
                    raise
                wait = min(2 ** attempt, 30)
                log.warning("llm.retry", attempt=attempt, wait=wait, error=error_str[:200])
                time.sleep(wait)

        raise last_error or RuntimeError("LLM call failed after retries")

    def _generate_openai(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> BaseModel | None:
        schema = response_model.model_json_schema()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = ""
        try:
            if self._mode == "json_schema":
                raw = self._call_llm_openai(
                    messages, temperature, max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "schema": schema,
                            "strict": True,
                        },
                    },
                )
            elif self._mode == "json_object":
                messages_with_schema = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": (
                        f"{user}\n\nRESPOND ONLY WITH VALID JSON matching this schema:\n"
                        f"{json.dumps(schema, indent=2)}"
                    )},
                ]
                raw = self._call_llm_openai(
                    messages_with_schema, temperature, max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                messages_with_schema = [
                    {"role": "system", "content": (
                        f"{system}\n\nCRITICAL: You MUST respond with ONLY a valid JSON object. "
                        "No text before or after. No markdown. No explanation."
                    )},
                    {"role": "user", "content": (
                        f"{user}\n\nOUTPUT FORMAT — respond ONLY with valid JSON:\n"
                        f"{json.dumps(schema, indent=2)}"
                    )},
                ]
                raw = self._call_llm_openai(
                    messages_with_schema, temperature, max_tokens,
                )
        except Exception as exc:
            log.error("llm.call_failed", error=str(exc)[:300])
            raise

        # Clean artifacts
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        try:
            return response_model.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            log.warning(
                "llm.parse_failed",
                model=response_model.__name__,
                error=str(exc),
                raw_preview=raw[:300],
            )
            return None

    # ── Anthropic structured generation ─────────────────────────

    def _build_anthropic_user_content(self, user: str) -> str | list[dict[str, Any]]:
        """Split user message for optimal Anthropic prompt caching.

        If the message contains a SOURCE DOCUMENT block, the doc portion gets
        a cache_control breakpoint so retries on the same doc hit cache even
        when the instruction tail changes.
        """
        marker_end = "══════ END SOURCE DOCUMENT ══════"
        end_idx = user.find(marker_end)
        if end_idx < 0:
            return user

        end_idx += len(marker_end)
        doc_block = user[:end_idx]
        rest = user[end_idx:].strip()
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": doc_block, "cache_control": {"type": "ephemeral"}},
        ]
        if rest:
            blocks.append({"type": "text", "text": rest})
        return blocks

    def _generate_anthropic(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> BaseModel | None:
        # System with cache control (same for retries of same agent)
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        # User message: cache doc block separately
        user_content = self._build_anthropic_user_content(user)

        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                result = self._anthropic.messages.parse(
                    model=self._model,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    output_format=response_model,
                )
                if result.parsed_output is not None:
                    return result.parsed_output
                log.warning("llm.parse_returned_none", model=response_model.__name__)
                return None
            except KeyboardInterrupt:
                raise
            except _anthropic_mod.BadRequestError as exc:
                log.error("llm.bad_request", error=str(exc)[:300])
                raise
            except _anthropic_mod.AuthenticationError as exc:
                log.error("llm.auth_failed", error=str(exc)[:300])
                raise
            except _anthropic_mod.PermissionDeniedError as exc:
                log.error("llm.permission_denied", error=str(exc)[:300])
                raise
            except (
                _anthropic_mod.RateLimitError,
                _anthropic_mod.InternalServerError,
                _anthropic_mod.APIConnectionError,
            ) as exc:
                last_error = exc
                wait = min(2 ** attempt, 30)
                log.warning("llm.retry", attempt=attempt, wait=wait, error=str(exc)[:200])
                time.sleep(wait)
            except (ValidationError, json.JSONDecodeError) as exc:
                log.warning(
                    "llm.parse_failed",
                    model=response_model.__name__,
                    error=str(exc)[:300],
                )
                return None
            except Exception as exc:
                last_error = exc
                wait = min(2 ** attempt, 30)
                log.warning("llm.retry", attempt=attempt, wait=wait, error=str(exc)[:200])
                time.sleep(wait)

        log.error("llm.exhausted_retries", error=str(last_error)[:300])
        return None


# ═══════════════════════════════════════════════════════════════
# AGENT: DOCUMENT ANALYST
# ═══════════════════════════════════════════════════════════════

_ANALYST_SYSTEM = """\
You are a senior document analyst. Your task is to analyze a document and produce \
a structured analysis for generating synthetic training conversations.

Rules:
- Identify the document type precisely (normativa, circolare, wiki, report_bdi, json_data, comunicato, regolamento, altro).
- Extract 3-8 main topics and key entities.
- Assess complexity (low/medium/high) based on technical depth.
- Write a concise factual summary IN ITALIAN.
- Plan the number of conversation turns to generate per type. The total should \
be proportional to document length and richness:
  - short_qa: brief 1-2 turn exchanges (factual)
  - long_qa: 2-4 turn detailed discussions
  - expert_deep: 1-2 turn expert-level deep answers
  - negative: 1 turn where assistant correctly declines or says it lacks info
  - json_output: 1 turn where assistant returns structured JSON data
  - tool_call: 1 turn with realistic tool usage (only if domain-appropriate)

For short documents (<500 chars): total 2-4 turns.
For medium documents (500-3000 chars): total 4-8 turns.
For long documents (>3000 chars): total 6-12 turns.

If the document is JSON data, focus on json_output and short_qa turns.
"""


def analyze_document(llm: LLMClient, doc: ParsedDocument) -> DocumentAnalysis | None:
    content_preview = _truncate_clean(doc.content)
    tag = "JSON DATA" if doc.is_json else "DOCUMENT"

    user_prompt = f"Analyze this {tag} and produce the structured analysis.\n\n---\n{content_preview}\n---"

    result = llm.generate_structured(
        system=_ANALYST_SYSTEM,
        user=user_prompt,
        response_model=DocumentAnalysis,
        temperature=TEMP_ANALYSIS,
    )

    if result:
        log.info(
            "analyst.done",
            doc_type=result.doc_type,
            complexity=result.complexity,
            planned_turns=sum(result.turn_plan.values()),
        )
    else:
        log.warning("analyst.failed", doc_hash=doc.hash)

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT: TURN BUILDER
# ═══════════════════════════════════════════════════════════════

_TURN_BUILDER_SYSTEM_TEMPLATE = """\
You are an expert conversation designer creating Italian training data for a language \
model fine-tune.

═══════════════════════════════════════════════════════════════
RULE #1 — ABSOLUTE, OVERRIDES EVERYTHING ELSE:
The assistant in the generated conversation must ONLY use facts, data, numbers, \
dates, article references, percentages, and entities that are EXPLICITLY PRESENT \
in the SOURCE DOCUMENT provided below.

If a fact is NOT in the source document, the assistant MUST NOT say it.
- NO invented article numbers (e.g. "Art. 428a CRR II") unless the doc says it
- NO invented percentages (e.g. "5% del fatturato") unless the doc says it
- NO invented deadlines (e.g. "entro 4 ore") unless the doc says it
- NO invented court cases or citations unless the doc says it
- NO external knowledge — ONLY what is written in the source document

If the document does not contain enough information to answer a question fully, \
the assistant must give a shorter answer limited to what IS in the document, \
or say it does not have that specific information.

VIOLATION OF THIS RULE = UNUSABLE TRAINING DATA. Every single fact must be \
traceable to the source document.
═══════════════════════════════════════════════════════════════

OTHER RULES:
- ALL output in Italian. Correct grammar, precise terminology.
- The user message must sound natural and human — varied phrasing, not robotic.
- The assistant response must be natural, direct, no filler phrases.
- NEVER use: "Certamente!", "Ottima domanda!", "Certo!", "Ecco...", emoji.
- NEVER use: "Posso dirti che...", "In merito alla tua domanda...".
- Calibrate response length to question complexity.
- Rephrase and synthesize from the source — do NOT copy-paste verbatim.

DIVERSITY REQUIREMENTS:
- The system_prompt field MUST be unique: different persona, different framing each time.
- The user question MUST approach the topic from a DIFFERENT ANGLE each time.
- User register must vary: formal/informal, expert/novice, terse/verbose.

TURN TYPE: {turn_type}
{turn_type_instructions}

Document type: {doc_type}
Document topics: {topics}
"""

_TURN_TYPE_INSTRUCTIONS: dict[str, str] = {
    TurnType.SHORT_QA: """\
Generate a SHORT conversation: 1 user question + 1 concise assistant answer (2-5 sentences).
The question should be direct and specific. The answer factual and to the point.""",
    TurnType.LONG_QA: """\
Generate a MULTI-TURN conversation: 2-4 exchanges (user/assistant pairs).
Start with a broad question, then follow up with specifics.
Responses should be thorough but not bloated.""",
    TurnType.EXPERT_DEEP: """\
Generate 1 user question + 1 EXPERT-LEVEL detailed response.
The response should be comprehensive (15-40 sentences), well-structured with \
clear paragraphs, precise terminology, and deep analysis.
The user asks something that requires genuine expertise to answer well.""",
    TurnType.NEGATIVE: """\
Generate a conversation where the assistant CORRECTLY declines or states \
it cannot answer. Scenarios:
- User asks something outside the provided context
- User asks for speculation or prediction the model cannot make
- User requests something inappropriate or impossible
The assistant must decline naturally, without being preachy or condescending, \
and WITHOUT offering an alternative ("pero posso dirti che...").""",
    TurnType.JSON_OUTPUT: """\
Generate a conversation where the user EXPLICITLY asks for data in JSON format. \
The user message MUST contain the word "json" or "strutturato" or "formato strutturato". \
Examples of valid user messages:
- "Puoi darmi le scadenze in formato json?"
- "Mi servono i dati strutturati delle categorie NPL"
- "Restituiscimi un json con i requisiti patrimoniali"

The assistant responds with CLEAN JSON — no markdown wrappers, no ```json blocks, \
no explanatory text before or after the JSON.
The JSON must be valid, well-structured, and derived from the document context.

CRITICAL: If the user does NOT explicitly ask for JSON/structured data, the turn is INVALID.""",
    TurnType.TOOL_CALL: """\
Generate a conversation with TOOL USAGE following ChatML conventions.
The assistant must use <tool_call> tags for function invocations.
Tool responses appear as user messages with <tool_response> tags.
The assistant must NOT invent tool results — the conversation includes realistic \
tool responses that the assistant then interprets.
The conversation should have: user question -> assistant tool call -> tool response -> assistant final answer.""",
}


def _build_tools_xml(tools: list[dict[str, Any]]) -> str:
    tools_json = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return f"""You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def generate_turn(
    llm: LLMClient,
    doc: ParsedDocument,
    analysis: DocumentAnalysis,
    turn_type: TurnType,
    previous_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    content_preview = _truncate_clean(doc.content)
    topics_str = ", ".join(analysis.topics)

    system = _TURN_BUILDER_SYSTEM_TEMPLATE.format(
        turn_type=turn_type.value,
        turn_type_instructions=_TURN_TYPE_INSTRUCTIONS[turn_type],
        doc_type=analysis.doc_type,
        topics=topics_str,
    )

    user_prompt = (
        f"══════ SOURCE DOCUMENT (this is your ONLY source of truth) ══════\n"
        f"{content_preview}\n"
        f"══════ END SOURCE DOCUMENT ══════\n\n"
        f"Document type: {analysis.doc_type}\n"
        f"Summary: {analysis.summary}\n"
        f"Key entities: {', '.join(analysis.key_entities)}\n\n"
        f"Generate ONE conversation where the assistant answers ONLY using facts "
        f"from the source document above. Every claim, number, date, and reference "
        f"in the assistant's response must be traceable to the text above."
    )

    # Inject anti-repetition history: show what was already generated
    if previous_turns:
        history_lines: list[str] = []
        for i, prev in enumerate(previous_turns, 1):
            prev_questions = _extract_user_questions(prev)
            prev_system = ""
            for m in prev.get("messages", []):
                if m["role"] == "system":
                    prev_system = m["content"][:100]
                    break
            if prev_questions:
                history_lines.append(
                    f"  {i}. User asked: \"{prev_questions[0][:120]}\" | System: \"{prev_system}...\""
                )
        if history_lines:
            history_block = "\n".join(history_lines)
            user_prompt += (
                f"\n\n⚠️ ALREADY GENERATED for this document (DO NOT repeat similar "
                f"questions, system prompts, or response structures):\n{history_block}\n"
                f"Generate something COMPLETELY DIFFERENT — different angle, "
                f"different question style, different system prompt persona."
            )

    if turn_type == TurnType.TOOL_CALL:
        selected_tools = random.sample(DOMAIN_TOOLS, min(3, len(DOMAIN_TOOLS)))
        result = llm.generate_structured(
            system=system,
            user=user_prompt + f"\n\nAvailable tools:\n{json.dumps(selected_tools, ensure_ascii=False, indent=2)}",
            response_model=ToolCallConversation,
            temperature=TEMP_GENERATION,
            max_tokens=4096,
        )
        if not result:
            return None
        return _format_tool_call_turn(result, selected_tools)
    else:
        result = llm.generate_structured(
            system=system,
            user=user_prompt,
            response_model=GeneratedConversation,
            temperature=TEMP_GENERATION,
            max_tokens=4096,
        )
        if not result:
            return None
        return _format_standard_turn(result, turn_type)


def _format_standard_turn(
    conv: GeneratedConversation, turn_type: TurnType
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": conv.system_prompt}
    ]
    for msg in conv.messages:
        role = msg.role
        if role not in ("user", "assistant"):
            role = "user" if role == "tool" else "assistant"
        messages.append({"role": role, "content": msg.content})
    return {"messages": messages, "metadata": {"turn_type": turn_type.value}}


def _format_tool_call_turn(
    conv: ToolCallConversation,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    system_content = conv.system_prompt
    if "<tools>" not in system_content:
        tools_block = _build_tools_xml(tools)
        system_content = f"{system_content}\n\n# Tools\n\n{tools_block}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]
    for msg in conv.messages:
        role = msg.role
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": msg.content})
    return {"messages": messages, "metadata": {"turn_type": TurnType.TOOL_CALL.value}}


# ═══════════════════════════════════════════════════════════════
# AGENT: QUALITY VALIDATOR
# ═══════════════════════════════════════════════════════════════

_VALIDATOR_SYSTEM = """\
You are a strict quality validator for Italian LLM training data. \
Evaluate the following conversation against these criteria:

ACCEPT if ALL are true:
1. User turns sound natural and human — not robotic or formulaic
2. Assistant turns are natural, direct, no filler phrases
3. Length is calibrated to complexity (short answer for simple Q, long for complex)
4. If JSON is present, it is clean — no markdown wrappers, no ```json blocks
5. Italian is correct with precise terminology
6. No "Certamente!", "Ottima domanda!", "Certo!", emoji
7. No "pero posso dirti che..." pattern in negatives
8. No copy-paste from context without reworking
9. Tool calls (if any) follow <tool_call>/<tool_response> format correctly
10. No invented information beyond what the context supports
11. The conversation is COMPLETE: has at least one user question AND one assistant answer
12. If previous turns are listed, this turn asks a GENUINELY DIFFERENT question \
(different angle, not just rephrased)

REJECT if ANY of these are true:
- Robotic or template-like language
- English text in user/assistant turns (system can be English)
- Response is generic and adds no value
- Response length is wildly wrong for the question
- JSON wrapped in markdown code blocks
- Missing assistant response or incomplete conversation
- Question is essentially the same as a previously accepted question (even if reworded)
- System prompt is nearly identical to a previous one

Score 1-10 where 7+ = accept, <7 = reject. Be strict: when in doubt, reject.
"""


def validate_turn(
    llm: LLMClient,
    turn: dict[str, Any],
    doc_summary: str,
    doc_history: list[dict[str, Any]] | None = None,
) -> ValidationResult | None:
    conversation_text = "\n".join(
        f"[{m['role'].upper()}] {m['content'][:500]}"
        for m in turn["messages"]
    )

    history_context = ""
    if doc_history:
        prev_questions = []
        for prev in doc_history:
            for m in prev.get("messages", []):
                if m["role"] == "user":
                    prev_questions.append(m["content"][:150])
                    break
        if prev_questions:
            history_context = (
                f"\n\nPREVIOUSLY ACCEPTED turns for this document had these user questions:\n"
                + "\n".join(f"  - \"{q}\"" for q in prev_questions)
                + "\nREJECT if the new conversation asks essentially the same thing or follows the same pattern."
            )

    user_prompt = (
        f"Validate this conversation:\n\n{conversation_text}\n\n"
        f"Original document summary: {doc_summary}"
        f"{history_context}"
    )
    return llm.generate_structured(
        system=_VALIDATOR_SYSTEM,
        user=user_prompt,
        response_model=ValidationResult,
        temperature=TEMP_VALIDATION,
        max_tokens=512,
    )


# ═══════════════════════════════════════════════════════════════
# FINAL GUARDRAIL — Source-only grounding agent
# ═══════════════════════════════════════════════════════════════

_GROUNDING_AGENT_SYSTEM = """\
You are a strict fact-checking auditor. You receive TWO inputs:

1. A SOURCE DOCUMENT — this is the ONLY source of truth.
2. A GENERATED CONVERSATION between a user and an assistant.

IMPORTANT: The source document may be TRUNCATED (cut off mid-sentence). This is \
normal and expected — it does NOT mean the document is corrupted. Ignore any \
truncation artifacts. Focus ONLY on the content that IS present.

YOUR SOLE JOB: For each specific factual claim in the assistant's responses, \
check if that fact appears in the source document text provided.

Facts to check: article numbers, law references, regulation numbers, court cases, \
percentages, dates, deadlines, monetary amounts, entity names, specific procedures.

ACCEPT if:
- Every specific fact in the assistant's messages can be found in the source text.
- The assistant paraphrases or summarizes content from the source. This is fine.
- The assistant uses general knowledge phrasing without specific claims. This is fine.

REJECT ONLY if:
- The assistant cites a SPECIFIC article, law, regulation, percentage, deadline, \
or court case that DOES NOT appear anywhere in the source document text.
- Example: source says nothing about "Art. 428a" but assistant mentions "Art. 428a".

DO NOT reject because:
- The document is truncated or seems incomplete — that is normal.
- The assistant's answer is shorter than expected.
- The assistant rephrases information from the source.
- General statements without specific verifiable claims.

When in doubt about a general statement, ACCEPT. \
When in doubt about a specific citation or number, REJECT.
"""


class GroundingResult(BaseModel):
    """Result of the final grounding check."""

    accepted: bool = Field(description="True if all facts are grounded in source")
    reason: str = Field(description="Explanation — cite the specific ungrounded fact if rejecting")


def grounding_agent_check(
    llm: LLMClient,
    turn: dict[str, Any],
    source_doc: str,
) -> GroundingResult | None:
    """Final guardrail: LLM checks conversation against source document only."""
    conversation_text = "\n".join(
        f"[{m['role'].upper()}] {m['content']}"
        for m in turn["messages"]
    )

    # Truncate source to fit context — but give it as much as possible
    source_preview = _truncate_clean(source_doc)

    user_prompt = (
        f"══════ SOURCE DOCUMENT (may be truncated — this is normal) ══════\n"
        f"{source_preview}\n"
        f"══════ END SOURCE DOCUMENT ══════\n\n"
        f"══════ GENERATED CONVERSATION ══════\n"
        f"{conversation_text}\n"
        f"══════ END CONVERSATION ══════\n\n"
        f"Check ONLY: does the assistant cite any specific article, law, regulation, "
        f"percentage, deadline, or court case that is NOT present in the source text above? "
        f"Ignore document truncation. Accept general statements without specific citations."
    )

    return llm.generate_structured(
        system=_GROUNDING_AGENT_SYSTEM,
        user=user_prompt,
        response_model=GroundingResult,
        temperature=TEMP_VALIDATION,
        max_tokens=512,
    )


# ═══════════════════════════════════════════════════════════════
# DEDUPLICATION — Exact + Fuzzy
# ═══════════════════════════════════════════════════════════════

_STRIP_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = _STRIP_PUNCT.sub("", text.lower())
    return " ".join(text.split())


def _extract_user_questions(turn: dict[str, Any]) -> list[str]:
    """Extract all user messages from a turn (for dedup comparison)."""
    return [
        msg["content"]
        for msg in turn.get("messages", [])
        if msg["role"] == "user"
    ]


def _word_unigrams(text: str) -> set[str]:
    """Extract word-level unigrams from normalized text."""
    return set(_normalize_text(text).split())


def _word_bigrams(text: str) -> set[tuple[str, str]]:
    """Extract word-level bigrams from normalized text."""
    words = _normalize_text(text).split()
    if len(words) < 2:
        return {(words[0], "") if words else ("", "")}
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Similarity score: max of unigram and bigram Jaccard.

    Unigrams catch word-level overlap in short phrases.
    Bigrams catch phrase-level patterns in longer text.
    Taking the max gives robust dedup for both cases.
    """
    uni_sim = _jaccard(_word_unigrams(text_a), _word_unigrams(text_b))
    bi_sim = _jaccard(_word_bigrams(text_a), _word_bigrams(text_b))
    return max(uni_sim, bi_sim)


def compute_turn_hash(turn: dict[str, Any]) -> str:
    """Exact MD5 hash of non-system messages."""
    parts: list[str] = []
    for msg in turn.get("messages", []):
        if msg["role"] != "system":
            parts.append(f"{msg['role']}:{msg['content']}")
    fingerprint = "|".join(parts)
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()


def compute_normalized_hash(turn: dict[str, Any]) -> str:
    """Normalized MD5: lowercase, no punctuation. Catches trivial rephrasings."""
    parts: list[str] = []
    for msg in turn.get("messages", []):
        if msg["role"] != "system":
            parts.append(f"{msg['role']}:{_normalize_text(msg['content'])}")
    fingerprint = "|".join(parts)
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()


def is_fuzzy_duplicate(
    new_turn: dict[str, Any],
    existing_turns: list[dict[str, Any]],
    threshold: float = 0.55,
) -> bool:
    """Check if new_turn is semantically too similar to any existing turn.

    Compares user questions via Jaccard bigram similarity.
    Threshold 0.55 catches "scadenze dora" vs "scadenze DORA per le banche".
    """
    new_questions = _extract_user_questions(new_turn)
    if not new_questions:
        return False

    new_q_joined = " ".join(new_questions)

    for existing in existing_turns:
        existing_questions = _extract_user_questions(existing)
        if not existing_questions:
            continue
        existing_q_joined = " ".join(existing_questions)
        sim = jaccard_similarity(new_q_joined, existing_q_joined)
        if sim >= threshold:
            return True
    return False


def has_valid_structure(turn: dict[str, Any]) -> tuple[bool, str]:
    """Validate turn has correct ChatML structure.

    Returns (is_valid, reason).
    """
    messages = turn.get("messages", [])

    if len(messages) < 2:
        return False, "too_few_messages"

    # Must have at least one assistant message with content
    assistant_msgs = [
        m for m in messages
        if m.get("role") == "assistant" and m.get("content", "").strip()
    ]
    if not assistant_msgs:
        return False, "no_assistant_response"

    # Must have at least one user message
    user_msgs = [
        m for m in messages
        if m.get("role") == "user" and m.get("content", "").strip()
    ]
    if not user_msgs:
        return False, "no_user_message"

    # Roles must alternate correctly (after system)
    non_system = [m for m in messages if m["role"] != "system"]
    if non_system and non_system[0]["role"] != "user":
        return False, "must_start_with_user"

    # Check minimum content length
    for m in assistant_msgs:
        if len(m["content"].strip()) < 10:
            return False, "assistant_response_too_short"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════
# RULE-BASED VALIDATOR — Deterministic, runs before LLM validator
# ═══════════════════════════════════════════════════════════════

# Phrases that NEVER belong in quality training data
_BANNED_PHRASES: list[str] = [
    "certamente!",
    "certamente,",
    "ottima domanda",
    "bella domanda",
    "buona domanda",
    "certo!",
    "certo,",
    "sicuramente!",
    "con piacere",
    "ecco il json",
    "ecco il risultato",
    "ecco la risposta",
    "ecco le informazioni",
    "posso dirti che",
    "però posso",
    "purtroppo non posso, ma",
    "non posso aiutarti, però",
    "come assistente",
    "in qualità di assistente",
    "in quanto assistente",
    "come ia",
    "come intelligenza artificiale",
    "sono un modello",
    "sono un'intelligenza",
]

# JSON wrapper patterns that indicate unclean JSON output
_JSON_WRAPPER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```json", re.IGNORECASE),
    re.compile(r"```\s*\{"),
    re.compile(r"ecco.*?:\s*\{", re.IGNORECASE),
    re.compile(r"il json.*?:\s*\{", re.IGNORECASE),
    re.compile(r"risultato.*?:\s*\{", re.IGNORECASE),
]

# Simple English detection: common English words that shouldn't appear in Italian answers
_ENGLISH_MARKERS: list[str] = [
    " the ", " is ", " are ", " this ", " that ", " with ", " have ",
    " from ", " will ", " would ", " should ", " could ", " which ",
    " here is ", " here are ", " below ", " following ", " however ",
    " therefore ", " furthermore ", " additionally ", " please note ",
]


def rule_based_validate(
    turn: dict[str, Any],
    turn_type: TurnType,
    doc_history: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Deterministic quality checks that don't depend on LLM judgment.

    Returns (passed, failure_reason). Runs BEFORE the LLM validator.
    These checks are absolute — no scoring, just pass/fail.
    """
    messages = turn.get("messages", [])

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    system_msgs = [m for m in messages if m["role"] == "system"]

    # ── Check 1: Banned phrases in assistant messages ──
    for msg in assistant_msgs:
        content_lower = msg["content"].lower()
        for phrase in _BANNED_PHRASES:
            if phrase in content_lower:
                return False, f"banned_phrase:{phrase}"

    # ── Check 2: JSON output type must be clean JSON ──
    if turn_type == TurnType.JSON_OUTPUT:
        for msg in assistant_msgs:
            content = msg["content"].strip()
            # Check for markdown wrappers
            for pattern in _JSON_WRAPPER_PATTERNS:
                if pattern.search(content):
                    return False, "json_has_wrapper"
            # The entire assistant response should be valid JSON
            # (allow a brief intro sentence + JSON, but the JSON must be parseable)
            json_candidate = content
            # Try to extract JSON if there's text before it
            brace_pos = content.find("{")
            bracket_pos = content.find("[")
            start = -1
            if brace_pos >= 0 and bracket_pos >= 0:
                start = min(brace_pos, bracket_pos)
            elif brace_pos >= 0:
                start = brace_pos
            elif bracket_pos >= 0:
                start = bracket_pos

            if start == -1:
                return False, "json_output_no_json_found"
            if start > 0:
                # There's text before the JSON — that's a wrapper
                preamble = content[:start].strip()
                if len(preamble) > 5:  # more than trivial whitespace
                    return False, "json_output_has_preamble_text"
            json_candidate = content[start:]
            try:
                json.loads(json_candidate)
            except json.JSONDecodeError:
                # Try reverse: find last } or ]
                for end_char in ("}", "]"):
                    last_pos = json_candidate.rfind(end_char)
                    if last_pos >= 0:
                        try:
                            json.loads(json_candidate[: last_pos + 1])
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    return False, "json_output_invalid_json"

    # ── Check 3: English detection in user/assistant messages ──
    # Count English markers across ALL non-system messages combined
    total_english_hits = 0
    for msg in user_msgs + assistant_msgs:
        content_lower = " " + msg["content"].lower() + " "
        total_english_hits += sum(1 for marker in _ENGLISH_MARKERS if marker in content_lower)
    # Allow 1-2 stray English words (loanwords like "framework"), reject if 3+
    if total_english_hits >= 3:
        return False, f"english_detected:hits={total_english_hits}"

    # ── Check 4: System prompt similarity to previous turns ──
    if system_msgs and doc_history:
        new_system = system_msgs[0]["content"]
        for prev in doc_history:
            for prev_msg in prev.get("messages", []):
                if prev_msg["role"] == "system":
                    sim = jaccard_similarity(new_system, prev_msg["content"])
                    if sim > 0.70:
                        return False, f"system_prompt_too_similar:sim={sim:.2f}"
                    break

    # ── Check 5: User question similarity to previous turns ──
    if user_msgs and doc_history:
        new_first_q = user_msgs[0]["content"]
        for prev in doc_history:
            prev_users = [m for m in prev.get("messages", []) if m["role"] == "user"]
            if prev_users:
                sim = jaccard_similarity(new_first_q, prev_users[0]["content"])
                if sim > 0.50:
                    return False, f"user_question_too_similar:sim={sim:.2f}"

    # ── Check 6: Assistant response length sanity ──
    for msg in assistant_msgs:
        content = msg["content"].strip()
        # For short_qa, response shouldn't be a novel
        if turn_type == TurnType.SHORT_QA and len(content) > 2000:
            return False, "short_qa_response_too_long"
        # For expert_deep, response shouldn't be a one-liner
        if turn_type == TurnType.EXPERT_DEEP and len(content) < 200:
            return False, "expert_deep_response_too_short"

    # ── Check 7: Tool call format validation ──
    if turn_type == TurnType.TOOL_CALL:
        has_tool_call = False
        has_tool_response = False
        for msg in messages:
            if "<tool_call>" in msg.get("content", ""):
                has_tool_call = True
            if "<tool_response>" in msg.get("content", ""):
                has_tool_response = True
        if not has_tool_call:
            return False, "tool_call_missing_tool_call_tag"
        if not has_tool_response:
            return False, "tool_call_missing_tool_response_tag"
        # The tool_call must contain valid JSON
        for msg in assistant_msgs:
            tc_match = re.search(
                r"<tool_call>\s*(.*?)\s*</tool_call>",
                msg["content"],
                re.DOTALL,
            )
            if tc_match:
                try:
                    json.loads(tc_match.group(1))
                except json.JSONDecodeError:
                    return False, "tool_call_invalid_json"

    # ── Check 8: Emoji detection ──
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001f900-\U0001f9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+",
        flags=re.UNICODE,
    )
    for msg in assistant_msgs + user_msgs:
        if emoji_pattern.search(msg["content"]):
            return False, "contains_emoji"

    # ── Check 9: Role alternation (no two user or two assistant in a row) ──
    non_system = [m for m in messages if m["role"] != "system"]
    for i in range(1, len(non_system)):
        curr_role = non_system[i]["role"]
        prev_role = non_system[i - 1]["role"]
        # Allow user→user only if second is <tool_response>
        if curr_role == prev_role:
            if curr_role == "user" and "<tool_response>" in non_system[i].get("content", ""):
                continue
            return False, f"role_alternation_broken:consecutive_{curr_role}"

    # ── Check 10: JSON response coherence ──
    # ANY assistant message that is pure valid JSON must be preceded by a user
    # message that explicitly asks for JSON/structured data. This applies to ALL
    # turn types including json_output — if the model generates a "normal" user
    # question and then responds with JSON, that teaches wrong behavior.
    _JSON_REQUEST_KEYWORDS = {
        "json", "strutturato", "structured", "formato json", "in json",
        "formato strutturato", "dati strutturati",
    }
    for i, msg in enumerate(non_system):
        if msg["role"] != "assistant":
            continue
        content = msg["content"].strip()
        # Check if this looks like a pure JSON response
        if (content.startswith("{") and content.endswith("}")) or \
           (content.startswith("[") and content.endswith("]")):
            try:
                json.loads(content)
                # It IS valid JSON. Did the preceding user ask for it?
                preceding_user_text = ""
                for j in range(i - 1, -1, -1):
                    if non_system[j]["role"] == "user":
                        preceding_user_text = non_system[j]["content"].lower()
                        break
                user_asked_json = any(
                    kw in preceding_user_text for kw in _JSON_REQUEST_KEYWORDS
                )
                if not user_asked_json:
                    return False, "unsolicited_json_response"
            except (json.JSONDecodeError, ValueError):
                pass  # Not valid JSON, that's fine

    return True, "ok"


# ═══════════════════════════════════════════════════════════════
# GROUNDING CHECK — Verify assistant facts exist in source doc
# ═══════════════════════════════════════════════════════════════

# Patterns that extract specific factual claims from text
_GROUNDING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Article references: "Art. 17", "articolo 428a", "art. 74, par. 1"
    ("article", re.compile(
        r"(?:art(?:icolo|\.)\s*\d+[a-z]?(?:\s*,?\s*(?:par|comma|lett)\.\s*\d+)?)",
        re.IGNORECASE,
    )),
    # EU regulations: "Reg. UE 2022/2554", "Regolamento UE 2018/389"
    ("regulation", re.compile(
        r"(?:reg(?:olamento)?\.?\s*(?:UE|CE|delegato)\s*\d{4}/\d+)",
        re.IGNORECASE,
    )),
    # Italian laws: "D.Lgs. 231/2007", "L. 262/2005"
    ("law", re.compile(
        r"(?:D\.?\s*Lgs\.?\s*\d+/\d{4}|L\.\s*\d+/\d{4})",
        re.IGNORECASE,
    )),
    # Circulars: "Circolare 285", "Circ. 285"
    ("circular", re.compile(
        r"(?:circ(?:olare)?\.?\s*(?:n\.?\s*)?\d+)",
        re.IGNORECASE,
    )),
    # Court cases: "sentenza C-485/21", "causa C-123/20"
    ("court_case", re.compile(
        r"(?:sentenza|causa)\s+C-\d+/\d+",
        re.IGNORECASE,
    )),
    # Specific percentages: "4,5%", "5%", "72%"
    ("percentage", re.compile(
        r"\d+(?:[.,]\d+)?\s*%",
    )),
    # Specific timeframes with numbers: "entro 4 ore", "entro 24 ore", "entro 30 giorni"
    ("timeframe", re.compile(
        r"entro\s+\d+\s+(?:ore|giorni|mesi|anni|settimane)",
        re.IGNORECASE,
    )),
    # Specific monetary amounts: "fino al 5%", "fino a 10.000 euro"
    ("amount", re.compile(
        r"fino\s+(?:al?\s+)?\d+(?:[.,]\d+)?\s*(?:%|euro|EUR|milion[ie]|miliard[ie])",
        re.IGNORECASE,
    )),
    # Monetary ranges: "da 10.000 a 50.000 euro", "da 5 a 10 milioni"
    ("amount_range", re.compile(
        r"da\s+\d+(?:[.,]\d+)?\s+a\s+\d+(?:[.,]\d+)?\s*(?:euro|EUR|milion[ie]|miliard[ie])",
        re.IGNORECASE,
    )),
]


def check_grounding(
    turn: dict[str, Any],
    source_doc: str,
) -> tuple[bool, str]:
    """Verify that specific factual claims in assistant responses are grounded
    in the source document.

    Extracts numbers, article refs, percentages, dates, legal citations from
    assistant messages and checks if they appear in the source text.

    Returns (grounded, reason).
    """
    assistant_text = " ".join(
        msg["content"]
        for msg in turn.get("messages", [])
        if msg["role"] == "assistant"
    )

    if not assistant_text.strip():
        return True, "ok"

    source_lower = source_doc.lower()
    source_normalized = re.sub(r"[^\w\s%/.,]", " ", source_lower)

    ungrounded: list[str] = []
    total_claims = 0

    for claim_type, pattern in _GROUNDING_PATTERNS:
        matches = pattern.findall(assistant_text)
        for match_text in matches:
            total_claims += 1
            match_lower = match_text.lower().strip()
            # Exact match in source
            if match_lower in source_lower:
                continue
            # Normalized match
            match_norm = re.sub(r"[^\w\s%/.,]", " ", match_lower)
            match_norm = " ".join(match_norm.split())
            if match_norm in source_normalized:
                continue
            # For numbers/percentages, try just the number as whole word
            numbers_in_match = re.findall(r"\d+(?:[.,]\d+)?", match_text)
            if numbers_in_match:
                all_found = True
                for n in numbers_in_match:
                    # Word boundary check: number must not be part of a larger number
                    n_pattern = re.compile(r"(?<!\d)" + re.escape(n) + r"(?!\d)")
                    if not n_pattern.search(source_doc):
                        all_found = False
                        break
                if all_found:
                    continue
            ungrounded.append(f"{claim_type}:{match_text.strip()}")

    if total_claims == 0:
        return True, "ok"

    # HARD REJECT: any ungrounded legal citation
    hard_types = {"article", "regulation", "law", "circular", "court_case"}
    hard_violations = [u for u in ungrounded if u.split(":")[0] in hard_types]
    if hard_violations:
        return False, f"ungrounded_citation:{hard_violations[0]}"

    # SOFT REJECT: >30% of numeric claims ungrounded
    if len(ungrounded) > 0 and (len(ungrounded) / total_claims) > 0.30:
        return False, (
            f"too_many_ungrounded:{len(ungrounded)}/{total_claims} "
            f"first={ungrounded[0]}"
        )

    return True, "ok"


# ═══════════════════════════════════════════════════════════════
# JSONL WRITER — Incremental, crash-safe
# ═══════════════════════════════════════════════════════════════


class JSONLWriter:
    """Append-only JSONL writer with flush-on-write for crash safety."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        output = {k: v for k, v in record.items() if k != "metadata"}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def count_lines(self) -> int:
        if not self._path.exists():
            return 0
        with open(self._path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def shuffle(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return
        random.shuffle(lines)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".jsonl.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.writelines(lines)
            os.replace(tmp_path, self._path)
            log.info("output.shuffled", total_lines=len(lines))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# ═══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR — Lazy, streaming
# ═══════════════════════════════════════════════════════════════


class Pipeline:
    """Main orchestrator: consumes a lazy Iterator[ParsedDocument].

    Never materializes the full document list in memory.
    """

    def __init__(
        self,
        llm: LLMClient,
        journal: JournalManager,
        writer: JSONLWriter,
        target_messages: int,
        skip_validation: bool = False,
    ) -> None:
        self._llm = llm
        self._journal = journal
        self._writer = writer
        self._target = target_messages
        self._skip_validation = skip_validation
        self._generated = journal.total_generated
        self._rejected = 0
        self._duplicates = 0
        self._docs_seen = 0
        self._consecutive_errors = 0
        self._max_consecutive_errors = int(
            os.getenv("MAX_CONSECUTIVE_ERRORS", "5")
        )

    @property
    def remaining(self) -> int:
        return max(0, self._target - self._generated)

    def run(self, documents: Iterator[ParsedDocument]) -> None:
        log.info(
            "pipeline.start",
            target=self._target,
            already_generated=self._generated,
            remaining=self.remaining,
        )

        if self.remaining == 0:
            log.info("pipeline.already_complete")
            return

        for doc in documents:
            if self.remaining <= 0:
                log.info("pipeline.target_reached")
                break

            self._docs_seen += 1

            if self._journal.is_doc_processed(doc.hash):
                continue

            log.info(
                "pipeline.processing_doc",
                doc=repr(doc),
                docs_seen=self._docs_seen,
                remaining=self.remaining,
            )

            try:
                self._process_document(doc)
                self._consecutive_errors = 0  # reset on success
            except KeyboardInterrupt:
                log.warning("pipeline.interrupted_by_user")
                raise
            except Exception as exc:
                self._consecutive_errors += 1
                log.error(
                    "pipeline.doc_error",
                    doc_hash=doc.hash[:8],
                    error=str(exc)[:300],
                    consecutive_errors=self._consecutive_errors,
                )
                if self._consecutive_errors >= self._max_consecutive_errors:
                    log.error(
                        "pipeline.circuit_breaker",
                        msg=(
                            f"Stopped after {self._consecutive_errors} consecutive "
                            f"errors. Check vLLM server."
                        ),
                    )
                    break
                continue

            self._journal.mark_doc_processed(doc.hash)

        log.info(
            "pipeline.complete",
            total_generated=self._generated,
            rejected=self._rejected,
            duplicates=self._duplicates,
            docs_seen=self._docs_seen,
        )

    def _process_document(self, doc: ParsedDocument) -> None:
        analysis = analyze_document(self._llm, doc)
        if not analysis:
            log.warning("pipeline.analysis_failed_skipping", doc_hash=doc.hash[:8])
            return

        turn_plan = analysis.turn_plan
        total_planned = sum(turn_plan.values())
        if total_planned == 0:
            log.warning("pipeline.empty_plan", doc_hash=doc.hash[:8])
            return

        scale = min(1.0, self.remaining / total_planned)

        # Per-document history: tracks generated turns for anti-repetition
        doc_history: list[dict[str, Any]] = []
        # Cap: max 2 turns of any single type per document
        max_per_type = 2

        for turn_type_str, count in turn_plan.items():
            if self.remaining <= 0:
                break
            try:
                turn_type = TurnType(turn_type_str)
            except ValueError:
                log.warning("pipeline.unknown_turn_type", value=turn_type_str)
                continue

            adjusted_count = max(1, int(count * scale)) if count > 0 else 0
            adjusted_count = min(adjusted_count, max_per_type)

            for _ in range(adjusted_count):
                if self.remaining <= 0:
                    break
                self._generate_single_turn(doc, analysis, turn_type, doc_history)

    def _generate_single_turn(
        self,
        doc: ParsedDocument,
        analysis: DocumentAnalysis,
        turn_type: TurnType,
        doc_history: list[dict[str, Any]],
    ) -> None:
        last_reason = "unknown"

        for attempt in range(MAX_RETRIES_PER_TURN):
            # Pass doc_history so builder avoids repeating questions/patterns
            turn = generate_turn(
                self._llm, doc, analysis, turn_type, doc_history
            )
            if not turn:
                last_reason = "llm_returned_none"
                log.info(
                    "pipeline.attempt_failed",
                    type=turn_type.value,
                    attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                    reason=last_reason,
                )
                continue

            # Structural validation: must have user + assistant with content
            is_valid, reason = has_valid_structure(turn)
            if not is_valid:
                last_reason = f"structure:{reason}"
                log.info(
                    "pipeline.attempt_failed",
                    type=turn_type.value,
                    attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                    reason=last_reason,
                )
                continue

            # Deterministic rule-based validation (catches what LLM misses)
            rules_ok, rule_reason = rule_based_validate(
                turn, turn_type, doc_history
            )
            if not rules_ok:
                self._rejected += 1
                last_reason = f"rule:{rule_reason}"
                log.info(
                    "pipeline.attempt_failed",
                    type=turn_type.value,
                    attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                    reason=last_reason,
                )
                continue

            # Grounding check: verify facts are in source document
            grounded, grounding_reason = check_grounding(turn, doc.content)
            if not grounded:
                self._rejected += 1
                last_reason = f"grounding:{grounding_reason}"
                log.info(
                    "pipeline.attempt_failed",
                    type=turn_type.value,
                    attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                    reason=last_reason,
                )
                continue

            # Exact dedup
            turn_hash = compute_turn_hash(turn)
            if self._journal.is_duplicate(turn_hash):
                self._duplicates += 1
                last_reason = "exact_duplicate"
                continue

            # Normalized dedup (catches trivial case changes)
            norm_hash = compute_normalized_hash(turn)
            if self._journal.is_duplicate(norm_hash):
                self._duplicates += 1
                last_reason = "normalized_duplicate"
                continue

            # Fuzzy dedup against doc history (catches semantic near-duplicates)
            if is_fuzzy_duplicate(turn, doc_history, threshold=0.55):
                self._duplicates += 1
                last_reason = "fuzzy_duplicate"
                log.info(
                    "pipeline.attempt_failed",
                    type=turn_type.value,
                    attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                    reason=last_reason,
                )
                continue

            # Quality validation
            if not self._skip_validation:
                validation = validate_turn(
                    self._llm, turn, analysis.summary, doc_history
                )
                if validation and not validation.accepted:
                    self._rejected += 1
                    last_reason = f"validator:score={validation.score}:{validation.reason[:80]}"
                    log.info(
                        "pipeline.attempt_failed",
                        type=turn_type.value,
                        attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                        reason=last_reason,
                    )
                    continue
                if validation:
                    log.debug("pipeline.validated", score=validation.score)

            # ── FINAL GUARDRAIL: LLM grounding agent ──
            # Sees ONLY source doc + chat — no summary, no history.
            # One invented fact = reject.
            if not self._skip_validation:
                grounding = grounding_agent_check(
                    self._llm, turn, doc.content
                )
                if grounding and not grounding.accepted:
                    self._rejected += 1
                    last_reason = f"grounding_agent:{grounding.reason[:100]}"
                    log.info(
                        "pipeline.attempt_failed",
                        type=turn_type.value,
                        attempt=f"{attempt + 1}/{MAX_RETRIES_PER_TURN}",
                        reason=last_reason,
                    )
                    continue
                if grounding:
                    log.debug("pipeline.grounding_passed")

            # All checks passed — write
            self._writer.append(turn)
            self._journal.register_turn(turn_hash)
            self._journal.register_turn(norm_hash)
            self._journal.increment_generated()
            self._generated += 1
            doc_history.append(turn)

            log.info(
                "pipeline.turn_saved",
                type=turn_type.value,
                total=self._generated,
                remaining=self.remaining,
            )
            return

        log.warning(
            "pipeline.exhausted_retries",
            turn_type=turn_type.value,
            doc_hash=doc.hash[:8],
            last_reason=last_reason,
        )


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


@app.command()
def generate(
    messages: int = typer.Option(
        100, "--messages", "-m",
        help="Number of conversation turns to generate.", min=1,
    ),
    dataset: Path = typer.Option(
        ..., "--dataset", "-d",
        help="Directory containing pre-training dataset files.",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True,
    ),
    output: Path = typer.Option(
        Path("output/dataset.jsonl"), "--output", "-o",
        help="Path for the output JSONL file.",
    ),
    provider: str = typer.Option(
        "", "--provider", "-p",
        help="LLM provider: 'openai' (also vLLM), 'anthropic', or '' for auto-detect.",
    ),
    model: str = typer.Option(
        "", "--model",
        help="Model name (overrides env vars).",
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation",
        help="Skip the quality validation agent (faster, less quality control).",
    ),
    skip_shuffle: bool = typer.Option(
        False, "--skip-shuffle",
        help="Skip the final shuffle of the output file.",
    ),
    clean: bool = typer.Option(
        False, "--clean",
        help="Delete existing output and journal, start fresh.",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed."),
) -> None:
    """Generate synthetic Italian conversation turns from pre-training documents."""
    random.seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Clean start: wipe previous output and journal
    if clean:
        journal_path = output.parent / ".journal.json"
        for f in [output, journal_path]:
            if f.exists():
                f.unlink()
                log.info("clean.deleted", file=str(f))

    log.info(
        "config",
        messages=messages,
        dataset=str(dataset),
        output=str(output),
        provider=provider or "(auto-detect)",
        model=model or "(from env)",
        skip_validation=skip_validation,
    )

    try:
        llm = LLMClient(provider=provider, model=model)
    except RuntimeError as exc:
        log.error("startup.llm_failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    journal = JournalManager(output.parent)
    writer = JSONLWriter(output)

    # Lazy stream — never loads full files into memory
    doc_stream = stream_dataset_directory(dataset)

    pipeline = Pipeline(
        llm=llm,
        journal=journal,
        writer=writer,
        target_messages=messages,
        skip_validation=skip_validation,
    )

    try:
        pipeline.run(doc_stream)
    except KeyboardInterrupt:
        log.warning("interrupted — partial output preserved", total=pipeline._generated)

    if not skip_shuffle:
        writer.shuffle()

    final_count = writer.count_lines()
    log.info("done", output=str(output), total_conversations=final_count)


@app.command()
def stats(
    output: Path = typer.Option(
        Path("output/dataset.jsonl"), "--output", "-o",
        help="Path to the output JSONL file.",
    ),
) -> None:
    """Show statistics about the generated dataset."""
    if not output.exists():
        log.error("stats.file_not_found", path=str(output))
        raise typer.Exit(code=1)

    total = 0
    role_counts: dict[str, int] = {}
    lengths: list[int] = []

    with open(output, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            msgs = record.get("messages", [])
            lengths.append(len(msgs))
            for msg in msgs:
                role = msg.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

    journal_path = output.parent / ".journal.json"
    journal_info = ""
    if journal_path.exists():
        with open(journal_path, encoding="utf-8") as f:
            j = json.load(f)
            journal_info = (
                f"  Docs processed: {len(j.get('processed_docs', []))}\n"
                f"  Unique hashes:  {len(j.get('turn_hashes', []))}"
            )

    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n{'=' * 50}")
    print(f"  Dataset: {output}")
    print(f"  Total conversations: {total}")
    print(f"  Avg messages/conv:   {avg_len:.1f}")
    print(f"  Role distribution:   {role_counts}")
    if journal_info:
        print(journal_info)
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    app()