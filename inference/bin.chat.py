"""
Streaming chat with your SFT-trained model — Claude Code style.

Usage:
  python chat.py --model ./checkpoints/skylar-100M-Chat/best
  python chat.py --model checkpoints_sft/best --system "Sei un pirata."
  python chat.py --model checkpoints_sft/best --json
"""

import argparse
import json
import sys
import time
import os
import torch
from tokenizers import Tokenizer, decoders

from models.decoder import NanoTransformer
from utils.chatML import encode_chatml

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

MAX_HISTORY_TURNS = 3  # 3 coppie user/assistant = 6 messaggi

console = Console()


# ─────────────────────────────────────────────────────────────
# STREAMING DECODE — thin wrapper over model.generate_streaming()
# ─────────────────────────────────────────────────────────────

def stream_response(model, input_ids, tokenizer, *,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2):
    """
    Decode token IDs from model.generate_streaming() into text chunks.

    No sampling logic here — that lives in model.py where it belongs.
    This only handles: token ID → text, partial UTF-8 buffering, EOS cleanup.

    Yields:
        (text_chunk: str, is_done: bool)
    """
    # Build EOS set from tokenizer special tokens
    eos_ids: list[int] = []
    for name in ("<|im_end|>", "<|endoftext|>", "<eos>"):   # include <eos> too (consistent with generate.py + eval suite)
        tid = tokenizer.token_to_id(name)
        if tid is not None:
            eos_ids.append(tid)

    generated_ids: list[int] = []
    yielded_len = 0

    for token_id in model.generate_streaming(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_ids,
    ):
        # EOS: flush buffer and stop
        if token_id in eos_ids:
            if generated_ids:
                full_text = tokenizer.decode(generated_ids)
                remaining = full_text[yielded_len:]
                if remaining:
                    yield remaining, False
            yield "", True
            return

        generated_ids.append(token_id)
        full_text = tokenizer.decode(generated_ids)

        # Hold back partial UTF-8 (emoji split across tokens)
        new_chunk = full_text[yielded_len:]
        if "\ufffd" in new_chunk:
            continue

        if new_chunk:
            yield new_chunk, False
            yielded_len = len(full_text)

    # Max tokens reached
    yield "", True


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(model_name, n_params, device, system_prompt, settings):
    """Print a styled header panel."""
    info_table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    info_table.add_column(style="dim")
    info_table.add_column(style="white")
    info_table.add_row("Model", model_name)
    info_table.add_row("Params", f"{n_params:,}")
    info_table.add_row("Device", device)
    sys_display = system_prompt[:60] + ("..." if len(system_prompt) > 60 else "")
    info_table.add_row("System", sys_display)
    info_table.add_row("Config", settings)

    console.print()
    console.print(Panel(
        info_table,
        title="[bold bright_cyan]◆ Skylar Chat[/bold bright_cyan] [dim]— Sophia AI NanoTransformer[/dim]",
        border_style="cyan",
        width=min(console.width, 80),
        padding=(1, 2),
    ))
    console.print(
        "  [dim]Commands: /quit  /clear  /system <prompt>  /config  /help[/dim]",
    )
    console.print()


def print_stats(n_tokens, elapsed, prompt_tokens):
    """Print generation stats after response."""
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    console.print(
        f"\n  [bright_black][{n_tokens} tokens · {elapsed:.1f}s · {tok_s:.1f} tok/s · "
        f"ctx {prompt_tokens}+{n_tokens}][/bright_black]\n"
    )


def print_help():
    """Print available commands."""
    help_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), border_style="dim")
    help_table.add_column("Command", style="cyan", min_width=20)
    help_table.add_column("Description")
    help_table.add_row("/quit", "Exit chat")
    help_table.add_row("/clear", "Clear conversation history")
    help_table.add_row("/system <prompt>", "Change system prompt (clears history)")
    help_table.add_row("/config", "Show current configuration")
    help_table.add_row("/reset", "Reset to default system prompt")
    help_table.add_row("/temp <value>", "Set temperature (0.0-2.0)")
    help_table.add_row("/topk <value>", "Set top_k (0=disabled)")
    help_table.add_row("/topp <value>", "Set top_p (0.0-1.0)")
    help_table.add_row("/rep <value>", "Set repetition penalty (1.0=off)")
    help_table.add_row("/max <value>", "Set max tokens")
    help_table.add_row("/help", "Show this help")
    console.print()
    console.print(help_table)
    console.print()


def print_config(temperature, top_k, top_p, rep_penalty, max_tokens, system_prompt, history):
    """Print current configuration."""
    cfg_table = Table(box=box.ROUNDED, border_style="dim", show_header=False, padding=(0, 1))
    cfg_table.add_column("Setting", style="dim")
    cfg_table.add_column("Value", style="white")
    cfg_table.add_row("temperature", str(temperature))
    cfg_table.add_row("top_k", str(top_k))
    cfg_table.add_row("top_p", str(top_p))
    cfg_table.add_row("rep_penalty", str(rep_penalty))
    cfg_table.add_row("max_tokens", str(max_tokens))
    cfg_table.add_row("system", system_prompt[:50] + "...")
    cfg_table.add_row("history", f"{len(history) // 2} turns")
    console.print()
    console.print(cfg_table)
    console.print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

SYSTEM_DEFAULT = (
    "Sei Skylar di Sophia AI, un assistente conversazionale in italiano. "
    "Parla in modo naturale e utile."
)


def main():
    parser = argparse.ArgumentParser(description="Skylar Chat — Streaming")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--system", type=str, default=SYSTEM_DEFAULT)
    parser.add_argument("--json", action="store_true", help="Force JSON output mode")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # ── Load model ──
    model = NanoTransformer.from_pretrained(args.model).to(device)
    model.eval()
    tokenizer = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()

    n_params = model.count_params()

    # ── State ──
    system_prompt = args.system
    if args.json:
        system_prompt += " Rispondi sempre in formato JSON valido."

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    rep_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    history = []

    # ── Header ──
    clear_screen()
    settings_str = f"temp={temperature} top_k={top_k} top_p={top_p} rep={rep_penalty}"
    print_header(args.model, n_params, device, system_prompt, settings_str)

    # ── Chat loop ──
    while True:
        try:
            user_input = console.input("  [bold green]❯[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n  [dim]Bye! 👋[/dim]\n")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            cmd_name = cmd[0]

            if cmd_name == "/quit":
                console.print("\n  [dim]Bye! 👋[/dim]\n")
                break

            elif cmd_name == "/clear":
                history = []
                clear_screen()
                settings_str = f"temp={temperature} top_k={top_k} top_p={top_p} rep={rep_penalty}"
                print_header(args.model, n_params, device, system_prompt, settings_str)
                continue

            elif cmd_name == "/help":
                print_help()
                continue

            elif cmd_name == "/system" and len(cmd) > 1:
                system_prompt = user_input[8:].strip()
                history = []
                console.print(f"  [green]✓[/green] System: [dim]{system_prompt[:60]}[/dim]\n")
                continue

            elif cmd_name == "/reset":
                system_prompt = SYSTEM_DEFAULT
                history = []
                console.print("  [green]✓[/green] System prompt resettato\n")
                continue

            elif cmd_name == "/config":
                print_config(temperature, top_k, top_p, rep_penalty, max_tokens, system_prompt, history)
                continue

            elif cmd_name == "/temp" and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    console.print(f"  [green]✓[/green] temperature = {temperature}\n")
                except ValueError:
                    console.print("  [red]✗[/red] Valore non valido\n")
                continue

            elif cmd_name == "/topk" and len(cmd) > 1:
                try:
                    top_k = int(cmd[1])
                    console.print(f"  [green]✓[/green] top_k = {top_k}\n")
                except ValueError:
                    console.print("  [red]✗[/red] Valore non valido\n")
                continue

            elif cmd_name == "/topp" and len(cmd) > 1:
                try:
                    top_p = float(cmd[1])
                    console.print(f"  [green]✓[/green] top_p = {top_p}\n")
                except ValueError:
                    console.print("  [red]✗[/red] Valore non valido\n")
                continue

            elif cmd_name == "/rep" and len(cmd) > 1:
                try:
                    rep_penalty = float(cmd[1])
                    console.print(f"  [green]✓[/green] repetition_penalty = {rep_penalty}\n")
                except ValueError:
                    console.print("  [red]✗[/red] Valore non valido\n")
                continue

            elif cmd_name == "/max" and len(cmd) > 1:
                try:
                    max_tokens = int(cmd[1])
                    console.print(f"  [green]✓[/green] max_tokens = {max_tokens}\n")
                except ValueError:
                    console.print("  [red]✗[/red] Valore non valido\n")
                continue

            else:
                console.print("  [red]✗[/red] Comando sconosciuto. Scrivi [cyan]/help[/cyan]\n")
                continue

        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

        # ── Build conversation ──
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        token_ids = encode_chatml(messages, tokenizer, add_generation_prompt=True)
        input_ids = torch.tensor([token_ids], device=device)

        # Truncate if needed — remove oldest turns
        while input_ids.shape[1] > model.config.max_seq_len - max_tokens and len(history) > 2:
            history = history[2:]
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_input})
            token_ids = encode_chatml(messages, tokenizer, add_generation_prompt=True)
            input_ids = torch.tensor([token_ids], device=device)

        prompt_tokens = len(token_ids)

        # ── Generate (streaming) ──
        # Use raw stdout for streaming tokens — Rich would buffer/reformat them
        console.print(f"\n  [bold cyan]◆[/bold cyan] ", end="")

        t0 = time.perf_counter()
        n_tokens = 0
        full_response = ""

        for chunk, is_done in stream_response(
                model, input_ids, tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
        ):
            if is_done:
                break
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_response += chunk
            n_tokens += 1

        elapsed = time.perf_counter() - t0
        full_response = full_response.strip()

        # JSON pretty-print
        if args.json and full_response:
            try:
                parsed = json.loads(full_response)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                sys.stdout.write(f"\r  ")
                console.print(f"[bold cyan]◆[/bold cyan] {formatted}", highlight=False)
            except json.JSONDecodeError:
                pass

        # Stats
        print_stats(n_tokens, elapsed, prompt_tokens)

        # Update history
        if full_response:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()