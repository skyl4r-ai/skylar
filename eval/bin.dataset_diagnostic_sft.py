#!/usr/bin/env python3

"""
Conversation dataset analyzer — extracts statistics and semantic biases from JSONL datasets.

Supports any HuggingFace tokenizer (local or hub) and standard conversation JSONL formats:
  - {"messages": [{"role": ..., "content": ...}, ...]}
  - {"conversations": [{"from": ..., "value": ...}, ...]}
  - {"prompt": ..., "completion": ...}

Usage:
    python dataset_analyze_sft.py ./data/distilled_sft_dataset.jsonl ./checkpoints/skylar-100M-Base
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationStats:
    """Aggregated statistics for the entire dataset."""

    total_conversations: int = 0
    total_tokens: int = 0
    total_turns: int = 0
    tokens_per_conversation: list[int] = field(default_factory=list)
    turns_per_conversation: list[int] = field(default_factory=list)
    role_token_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    role_turn_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    all_texts_by_role: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / max(self.total_conversations, 1)

    @property
    def median_tokens(self) -> float:
        s = sorted(self.tokens_per_conversation)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0

    @property
    def std_tokens(self) -> float:
        if len(self.tokens_per_conversation) < 2:
            return 0.0
        mean = self.avg_tokens
        variance = sum((t - mean) ** 2 for t in self.tokens_per_conversation) / len(
            self.tokens_per_conversation
        )
        return math.sqrt(variance)

    @property
    def avg_turns(self) -> float:
        return self.total_turns / max(self.total_conversations, 1)


@dataclass
class BiasReport:
    """Semantic bias analysis results."""

    sentiment_by_role: dict[str, dict[str, float]] = field(default_factory=dict)
    top_tfidf_terms_by_role: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    topic_clusters: list[dict[str, Any]] = field(default_factory=list)
    lexical_diversity_by_role: dict[str, float] = field(default_factory=dict)
    repetition_hotspots: list[tuple[str, int]] = field(default_factory=list)
    role_dominance: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str) -> Any:
    """Load a HuggingFace tokenizer from a local path or hub identifier.

    Args:
        tokenizer_path: Local directory or HuggingFace model identifier.

    Returns:
        A tokenizer instance with an `encode` method.

    Raises:
        SystemExit: If the tokenizer cannot be loaded.
    """
    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore[import-untyped]
    except ImportError:
        console.print(
            "[bold red]Error:[/] `transformers` is required. "
            "Install with: pip install transformers",
        )
        raise SystemExit(1)

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, trust_remote_code=True)
        console.print(f"[green]✓[/] Tokenizer loaded from [cyan]{tokenizer_path}[/]")
        return tokenizer
    except Exception as exc:
        console.print(f"[bold red]Error loading tokenizer:[/] {exc}")
        raise SystemExit(1)


def count_tokens(tokenizer: Any, text: str) -> int:
    """Count tokens for a given text string."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# JSONL parsing (supports multiple conversation formats)
# ---------------------------------------------------------------------------

def _normalize_role(role: str) -> str:
    """Map various role names to a canonical set."""
    role = role.strip().lower()
    mapping: dict[str, str] = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "ai": "assistant",
    }
    return mapping.get(role, role)


def _extract_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Extract a normalized list of {'role': ..., 'content': ...} from a record.

    Supports:
        - OpenAI / ShareGPT `messages` format
        - ShareGPT `conversations` format
        - Simple `prompt` / `completion` format
    """
    if "messages" in record:
        return [
            {"role": _normalize_role(m.get("role", "unknown")), "content": m.get("content", "")}
            for m in record["messages"]
        ]

    if "conversations" in record:
        return [
            {
                "role": _normalize_role(m.get("from", m.get("role", "unknown"))),
                "content": m.get("value", m.get("content", "")),
            }
            for m in record["conversations"]
        ]

    if "prompt" in record or "completion" in record:
        msgs: list[dict[str, str]] = []
        if "prompt" in record:
            msgs.append({"role": "user", "content": record["prompt"]})
        if "completion" in record:
            msgs.append({"role": "assistant", "content": record["completion"]})
        return msgs

    # Last resort: concatenate all string values
    text = " ".join(str(v) for v in record.values() if isinstance(v, str))
    if text.strip():
        return [{"role": "unknown", "content": text}]
    return []


def parse_jsonl(filepath: Path) -> list[list[dict[str, str]]]:
    """Parse a JSONL file into a list of conversations.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of conversations, each being a list of message dicts.

    Raises:
        SystemExit: If the file cannot be read or parsed.
    """
    if not filepath.exists():
        console.print(f"[bold red]Error:[/] File not found: {filepath}")
        raise SystemExit(1)

    conversations: list[list[dict[str, str]]] = []
    errors = 0

    with filepath.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                messages = _extract_messages(record)
                if messages:
                    conversations.append(messages)
            except json.JSONDecodeError:
                errors += 1
                if errors <= 5:
                    logger.warning("Skipping malformed JSON at line %d", line_no)

    if errors:
        console.print(f"[yellow]⚠[/] Skipped {errors} malformed line(s)")

    console.print(f"[green]✓[/] Parsed [cyan]{len(conversations)}[/] conversations from {filepath.name}")
    return conversations


# ---------------------------------------------------------------------------
# Statistics collection
# ---------------------------------------------------------------------------

def compute_stats(
    conversations: list[list[dict[str, str]]],
    tokenizer: Any,
) -> ConversationStats:
    """Compute aggregate conversation statistics.

    Args:
        conversations: Parsed conversation list.
        tokenizer: A tokenizer with an `encode` method.

    Returns:
        Populated ConversationStats object.
    """
    stats = ConversationStats()
    stats.total_conversations = len(conversations)

    with console.status("[bold cyan]Tokenizing conversations…"):
        for convo in conversations:
            conv_tokens = 0
            conv_turns = len(convo)
            for msg in convo:
                role = msg["role"]
                content = msg["content"]
                n_tok = count_tokens(tokenizer, content)
                conv_tokens += n_tok
                stats.role_token_counts[role] += n_tok
                stats.role_turn_counts[role] += 1
                stats.all_texts_by_role[role].append(content)

            stats.total_tokens += conv_tokens
            stats.total_turns += conv_turns
            stats.tokens_per_conversation.append(conv_tokens)
            stats.turns_per_conversation.append(conv_turns)

    return stats


# ---------------------------------------------------------------------------
# Semantic bias analysis
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: set[str] = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "happy", "joy", "brilliant", "perfect", "best", "beautiful", "awesome",
    "pleasant", "kind", "helpful", "positive", "success", "agree", "correct",
    "right", "sure", "yes", "absolutely", "certainly", "glad", "thank",
    "please", "appreciate", "impressive", "outstanding", "remarkable",
}
_NEGATIVE_WORDS: set[str] = {
    "bad", "terrible", "awful", "horrible", "hate", "sad", "angry", "wrong",
    "fail", "failure", "worst", "ugly", "poor", "negative", "disagree",
    "error", "mistake", "problem", "unfortunately", "sorry", "cannot",
    "impossible", "difficult", "danger", "dangerous", "risk", "harm",
    "concern", "warning", "never", "no", "not", "don't", "shouldn't",
}
_HEDGING_WORDS: set[str] = {
    "maybe", "perhaps", "possibly", "might", "could", "seems", "appears",
    "suggest", "likely", "unlikely", "approximately", "roughly", "generally",
    "sometimes", "often", "usually", "tend", "somewhat", "relatively",
    "arguably", "potentially", "presumably", "apparently", "allegedly",
}
_AUTHORITY_WORDS: set[str] = {
    "must", "shall", "always", "never", "definitely", "certainly", "clearly",
    "obviously", "undoubtedly", "absolutely", "exactly", "precisely",
    "fundamental", "essential", "critical", "crucial", "important",
    "necessary", "require", "mandatory", "imperative", "vital",
}

_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "up", "down", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "not", "no", "nor", "don't", "didn't", "won't",
    "also", "like", "get", "got", "make", "know", "think", "well",
    "one", "two", "even", "new", "want", "use", "way", "look", "first",
    "come", "see", "need", "back", "going", "much", "let", "really",
    "said", "say", "go", "time", "thing", "things", "still",
}

_WORD_RE = re.compile(r"[a-z]{2,}")


def _tokenize_text(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _word_set_ratio(tokens: list[str], word_set: set[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in word_set) / len(tokens)


def _compute_sentiment(texts: list[str]) -> dict[str, float]:
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(_tokenize_text(t))

    return {
        "positive_ratio": round(_word_set_ratio(all_tokens, _POSITIVE_WORDS), 4),
        "negative_ratio": round(_word_set_ratio(all_tokens, _NEGATIVE_WORDS), 4),
        "hedging_ratio": round(_word_set_ratio(all_tokens, _HEDGING_WORDS), 4),
        "authority_ratio": round(_word_set_ratio(all_tokens, _AUTHORITY_WORDS), 4),
    }


def _tfidf_terms(
    corpus_by_role: dict[str, list[str]],
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Compute TF-IDF per role, treating each role's concatenated texts as one document.

    This surfaces terms that are distinctively frequent in one role vs. others.
    """
    role_tokens: dict[str, Counter[str]] = {}
    for role, texts in corpus_by_role.items():
        counter: Counter[str] = Counter()
        for t in texts:
            words = [w for w in _tokenize_text(t) if w not in _STOPWORDS]
            counter.update(words)
        role_tokens[role] = counter

    # Document frequency: in how many roles does a term appear?
    all_terms: set[str] = set()
    for c in role_tokens.values():
        all_terms.update(c.keys())

    n_docs = len(role_tokens)
    df: dict[str, int] = {}
    for term in all_terms:
        df[term] = sum(1 for c in role_tokens.values() if term in c)

    result: dict[str, list[tuple[str, float]]] = {}
    for role, counter in role_tokens.items():
        total = sum(counter.values())
        if total == 0:
            result[role] = []
            continue

        scored: list[tuple[str, float]] = []
        for term, count in counter.items():
            tf = count / total
            idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1
            scored.append((term, round(tf * idf, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        result[role] = scored[:top_k]

    return result


def _lexical_diversity(texts: list[str]) -> float:
    """Type-token ratio (TTR) as a simple lexical diversity measure."""
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(_tokenize_text(t))
    if not all_tokens:
        return 0.0
    # Root TTR (Guiraud's index) to reduce length sensitivity
    types = len(set(all_tokens))
    return round(types / math.sqrt(len(all_tokens)), 4)


def _find_repetition_hotspots(
    texts: list[str],
    ngram_size: int = 3,
    top_k: int = 15,
) -> list[tuple[str, int]]:
    """Find most repeated n-grams across all texts (phrase-level repetition)."""
    ngram_counts: Counter[str] = Counter()
    for text in texts:
        words = _tokenize_text(text)
        for i in range(len(words) - ngram_size + 1):
            gram = " ".join(words[i : i + ngram_size])
            # Filter out pure stopword n-grams
            if all(w in _STOPWORDS for w in words[i : i + ngram_size]):
                continue
            ngram_counts[gram] += 1

    return ngram_counts.most_common(top_k)


def analyze_biases(stats: ConversationStats, top_k: int = 15) -> BiasReport:
    """Run semantic bias analysis on collected conversation data.

    Args:
        stats: Pre-computed conversation statistics.
        top_k: Number of top terms to surface per analysis.

    Returns:
        A populated BiasReport.
    """
    report = BiasReport()

    with console.status("[bold cyan]Analyzing semantic biases…"):
        # 1. Sentiment per role
        for role, texts in stats.all_texts_by_role.items():
            report.sentiment_by_role[role] = _compute_sentiment(texts)

        # 2. TF-IDF distinctive terms per role
        report.top_tfidf_terms_by_role = _tfidf_terms(stats.all_texts_by_role, top_k)

        # 3. Lexical diversity per role
        for role, texts in stats.all_texts_by_role.items():
            report.lexical_diversity_by_role[role] = _lexical_diversity(texts)

        # 4. Repetition hotspots (across ALL text)
        all_texts = [t for texts in stats.all_texts_by_role.values() for t in texts]
        report.repetition_hotspots = _find_repetition_hotspots(all_texts, ngram_size=3, top_k=top_k)

        # 5. Role dominance (token share per role)
        total = sum(stats.role_token_counts.values())
        if total > 0:
            for role, count in stats.role_token_counts.items():
                report.role_dominance[role] = round(count / total, 4)

    return report


# ---------------------------------------------------------------------------
# Rich output rendering
# ---------------------------------------------------------------------------

def _bar(value: float, max_width: int = 30) -> str:
    filled = int(value * max_width)
    return "█" * filled + "░" * (max_width - filled)


def render_stats(stats: ConversationStats) -> None:
    """Print conversation statistics as rich panels and tables."""
    # --- Overview panel ---
    overview = Table(show_header=False, box=None, padding=(0, 2))
    overview.add_column("Metric", style="bold white")
    overview.add_column("Value", style="cyan", justify="right")
    overview.add_row("Total conversations", f"{stats.total_conversations:,}")
    overview.add_row("Total tokens", f"{stats.total_tokens:,}")
    overview.add_row("Total turns", f"{stats.total_turns:,}")
    overview.add_row("Avg tokens / conversation", f"{stats.avg_tokens:,.1f}")
    overview.add_row("Median tokens / conversation", f"{stats.median_tokens:,.1f}")
    overview.add_row("Std dev tokens", f"{stats.std_tokens:,.1f}")
    overview.add_row("Avg turns / conversation", f"{stats.avg_turns:,.1f}")

    if stats.tokens_per_conversation:
        overview.add_row("Min tokens", f"{min(stats.tokens_per_conversation):,}")
        overview.add_row("Max tokens", f"{max(stats.tokens_per_conversation):,}")

    console.print(Panel(overview, title="[bold]📊 Dataset Overview", border_style="blue", box=box.ROUNDED))

    # --- Per-role breakdown ---
    role_table = Table(title="Token & Turn Distribution by Role", box=box.SIMPLE_HEAVY, border_style="dim")
    role_table.add_column("Role", style="bold magenta")
    role_table.add_column("Tokens", justify="right", style="cyan")
    role_table.add_column("% Tokens", justify="right")
    role_table.add_column("Turns", justify="right", style="cyan")

    total_tok = max(sum(stats.role_token_counts.values()), 1)
    for role in sorted(stats.role_token_counts):
        tok = stats.role_token_counts[role]
        pct = tok / total_tok
        role_table.add_row(role, f"{tok:,}", f"{pct:.1%}", f"{stats.role_turn_counts[role]:,}")

    console.print(role_table)

    # --- Token distribution histogram (text-based) ---
    if stats.tokens_per_conversation:
        _render_histogram(stats.tokens_per_conversation)


def _render_histogram(values: list[int], bins: int = 10) -> None:
    lo, hi = min(values), max(values)
    if lo == hi:
        return
    step = max((hi - lo) / bins, 1)
    bucket_counts: list[int] = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        bucket_counts[idx] += 1

    max_count = max(bucket_counts)
    bar_width = 25

    table = Table(title="Token Distribution", box=box.SIMPLE, border_style="dim", show_edge=False)
    table.add_column("Range", style="white", min_width=16)
    table.add_column("Count", justify="right", style="cyan", min_width=6)
    table.add_column("", min_width=bar_width + 2)

    for i, count in enumerate(bucket_counts):
        range_lo = int(lo + i * step)
        range_hi = int(lo + (i + 1) * step)
        filled = int((count / max(max_count, 1)) * bar_width)
        bar = "▓" * filled + "░" * (bar_width - filled)
        table.add_row(f"{range_lo:>7,} – {range_hi:>7,}", str(count), bar)

    console.print(table)


def render_biases(report: BiasReport) -> None:
    """Print bias analysis results as rich panels and tables."""

    # --- Sentiment bias ---
    sent_table = Table(title="Sentiment & Tone Bias by Role", box=box.SIMPLE_HEAVY, border_style="dim")
    sent_table.add_column("Role", style="bold magenta")
    sent_table.add_column("Positive", justify="right", style="green")
    sent_table.add_column("Negative", justify="right", style="red")
    sent_table.add_column("Hedging", justify="right", style="yellow")
    sent_table.add_column("Authority", justify="right", style="blue")

    for role in sorted(report.sentiment_by_role):
        s = report.sentiment_by_role[role]
        sent_table.add_row(
            role,
            f"{s['positive_ratio']:.2%}",
            f"{s['negative_ratio']:.2%}",
            f"{s['hedging_ratio']:.2%}",
            f"{s['authority_ratio']:.2%}",
        )

    console.print(Panel(sent_table, title="[bold]🎭 Sentiment & Tone Analysis", border_style="magenta"))

    # --- Role dominance ---
    if report.role_dominance:
        dom_table = Table(show_header=False, box=None, padding=(0, 1))
        dom_table.add_column("Role", style="bold magenta", min_width=12)
        dom_table.add_column("Share", min_width=35)
        dom_table.add_column("Pct", justify="right", style="cyan", min_width=6)

        for role in sorted(report.role_dominance, key=report.role_dominance.get, reverse=True):  # type: ignore[arg-type]
            pct = report.role_dominance[role]
            dom_table.add_row(role, _bar(pct), f"{pct:.1%}")

        console.print(Panel(dom_table, title="[bold]⚖️  Role Dominance (Token Share)", border_style="yellow"))

    # --- Lexical diversity ---
    div_table = Table(show_header=False, box=None, padding=(0, 2))
    div_table.add_column("Role", style="bold magenta")
    div_table.add_column("Guiraud's Index", justify="right", style="cyan")

    for role in sorted(report.lexical_diversity_by_role):
        div_table.add_row(role, f"{report.lexical_diversity_by_role[role]:.2f}")

    console.print(Panel(div_table, title="[bold]📖 Lexical Diversity (Guiraud's Index)", border_style="green"))

    # --- TF-IDF distinctive terms ---
    for role in sorted(report.top_tfidf_terms_by_role):
        terms = report.top_tfidf_terms_by_role[role]
        if not terms:
            continue

        tf_table = Table(box=box.SIMPLE, border_style="dim", show_edge=False)
        tf_table.add_column("Term", style="white")
        tf_table.add_column("TF-IDF", justify="right", style="cyan")
        tf_table.add_column("", min_width=20)

        max_score = terms[0][1] if terms else 1
        for term, score in terms:
            filled = int((score / max_score) * 20)
            bar = "▓" * filled + "░" * (20 - filled)
            tf_table.add_row(term, f"{score:.4f}", bar)

        console.print(
            Panel(tf_table, title=f"[bold]🔍 Distinctive Terms — [magenta]{role}[/]", border_style="cyan")
        )

    # --- Repetition hotspots ---
    if report.repetition_hotspots:
        rep_table = Table(box=box.SIMPLE, border_style="dim", show_edge=False)
        rep_table.add_column("Trigram", style="white")
        rep_table.add_column("Occurrences", justify="right", style="cyan")

        for gram, count in report.repetition_hotspots:
            rep_table.add_row(gram, f"{count:,}")

        console.print(
            Panel(rep_table, title="[bold]🔁 Repetition Hotspots (Top Trigrams)", border_style="red")
        )

    # --- Summary interpretation ---
    _render_interpretation(report)


def _render_interpretation(report: BiasReport) -> None:
    """Generate a human-readable bias summary."""
    findings: list[str] = []

    # Sentiment skew
    for role, sent in report.sentiment_by_role.items():
        pos, neg = sent["positive_ratio"], sent["negative_ratio"]
        if pos > 0 and neg > 0:
            ratio = pos / neg
            if ratio > 3:
                findings.append(
                    f"[magenta]{role}[/] shows strong [green]positive sentiment bias[/] "
                    f"(positive/negative ratio: {ratio:.1f}x)"
                )
            elif ratio < 0.5:
                findings.append(
                    f"[magenta]{role}[/] shows strong [red]negative sentiment bias[/] "
                    f"(positive/negative ratio: {ratio:.1f}x)"
                )

        if sent["hedging_ratio"] > 0.02:
            findings.append(
                f"[magenta]{role}[/] uses [yellow]high hedging language[/] ({sent['hedging_ratio']:.2%}) "
                "— may indicate uncertainty bias or over-cautious tone"
            )
        if sent["authority_ratio"] > 0.02:
            findings.append(
                f"[magenta]{role}[/] uses [blue]high authority language[/] ({sent['authority_ratio']:.2%}) "
                "— may indicate assertiveness or prescriptive bias"
            )

    # Dominance imbalance
    if report.role_dominance:
        values = list(report.role_dominance.values())
        if len(values) >= 2 and max(values) > 0.75:
            dominant = max(report.role_dominance, key=report.role_dominance.get)  # type: ignore[arg-type]
            findings.append(
                f"[magenta]{dominant}[/] dominates with [cyan]{report.role_dominance[dominant]:.1%}[/] "
                "of all tokens — significant role imbalance"
            )

    # Lexical diversity gap
    if len(report.lexical_diversity_by_role) >= 2:
        roles_div = sorted(report.lexical_diversity_by_role.items(), key=lambda x: x[1])
        lo_role, lo_val = roles_div[0]
        hi_role, hi_val = roles_div[-1]
        if hi_val > 0 and lo_val / hi_val < 0.7:
            findings.append(
                f"[magenta]{lo_role}[/] has significantly [red]lower lexical diversity[/] "
                f"({lo_val:.2f}) vs [magenta]{hi_role}[/] ({hi_val:.2f}) "
                "— may indicate repetitive or formulaic language"
            )

    if findings:
        text = Text()
        for i, f in enumerate(findings, 1):
            text.append(f"  {i}. ")
            text.append_text(Text.from_markup(f))
            text.append("\n")
        console.print(
            Panel(text, title="[bold]🧠 Bias Interpretation", border_style="bright_white", box=box.DOUBLE)
        )
    else:
        console.print(
            Panel(
                "[dim]No strong biases detected — dataset appears relatively balanced.",
                title="[bold]🧠 Bias Interpretation",
                border_style="bright_white",
            )
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    dataset_path: str = typer.Argument(..., help="Path to the JSONL dataset file"),
    tokenizer_path: str = typer.Argument(..., help="HuggingFace tokenizer path (local dir or hub id)"),
    top_k: int = typer.Option(15, "--top-k", "-k", help="Number of top terms per analysis"),
) -> None:
    """Analyze a JSONL conversation dataset for statistics and semantic biases."""
    console.print(
        Panel(
            "[bold white]Conversation Dataset Analyzer[/]\n"
            "[dim]Statistics · Sentiment · TF-IDF · Lexical Diversity · Bias Detection[/]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )
    console.print()

    filepath = Path(dataset_path)
    tokenizer = load_tokenizer(tokenizer_path)
    conversations = parse_jsonl(filepath)

    if not conversations:
        console.print("[bold red]No conversations found in dataset. Exiting.[/]")
        raise SystemExit(1)

    stats = compute_stats(conversations, tokenizer)
    console.print()
    render_stats(stats)

    console.print()
    report = analyze_biases(stats, top_k=top_k)
    render_biases(report)

    console.print(f"\n[dim]Analysis complete — {stats.total_conversations:,} conversations processed.[/]\n")


if __name__ == "__main__":
    app()