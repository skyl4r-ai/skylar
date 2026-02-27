"""
Chat dataset utilities for Supervised Fine-Tuning (SFT).

Supports:
  - ChatML format (same as Qwen, Mistral, OpenAI)
The key insight: a "chat model" is just a base model fine-tuned on
structured conversations. The structure is enforced by special tokens.
"""

import json


# ─────────────────────────────────────────────────────────────
# CHAT FORMAT (ChatML)
# ─────────────────────────────────────────────────────────────
#
# ChatML is the standard used by Qwen, Mistral, OpenAI, etc.
# It wraps each message in special tokens:
#
#   <|im_start|>system
#   You are a helpful assistant.<|im_end|>
#   <|im_start|>user
#   What is 2+2?<|im_end|>
#   <|im_start|>assistant
#   2+2 equals 4.<|im_end|>
#
# During training, we only compute loss on the ASSISTANT tokens.
# The model learns: "given this conversation so far, what should
# the assistant say next?"
# ─────────────────────────────────────────────────────────────

def get_chatml_ids(tokenizer):
    """Works with both special-token and sub-token tokenizers."""
    ims_id = tokenizer.token_to_id("<|im_start|>")
    if ims_id is not None:
        return [ims_id], [tokenizer.token_to_id("<|im_end|>")]
    else:
        return (
            tokenizer.encode("<|im_start|>", add_special_tokens=False).ids,
            tokenizer.encode("<|im_end|>", add_special_tokens=False).ids,
        )


def create_loss_mask(messages, tokenizer):
    """Loss mask for ChatML. Works with both special-token and sub-token tokenizers."""
    ims_ids, ime_ids = get_chatml_ids(tokenizer)

    all_token_ids = []
    all_labels = []

    for i, msg in enumerate(messages):
        is_assistant = (msg["role"] == "assistant")

        all_token_ids.extend(ims_ids)
        all_labels.extend([-100] * len(ims_ids))

        role_ids = tokenizer.encode(msg["role"] + "\n", add_special_tokens=False).ids
        all_token_ids.extend(role_ids)
        all_labels.extend([-100] * len(role_ids))

        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False).ids
        all_token_ids.extend(content_ids)
        if is_assistant:
            all_labels.extend(list(content_ids))
        else:
            all_labels.extend([-100] * len(content_ids))

        all_token_ids.extend(ime_ids)
        if is_assistant:
            all_labels.extend(list(ime_ids))
        else:
            all_labels.extend([-100] * len(ime_ids))

        if i < len(messages) - 1:
            sep_ids = tokenizer.encode("\n", add_special_tokens=False).ids
            all_token_ids.extend(sep_ids)
            all_labels.extend([-100] * len(sep_ids))

    return all_token_ids, all_labels


def encode_chatml(messages, tokenizer, add_generation_prompt=False):
    """Encode ChatML. Works with both special-token and sub-token tokenizers."""
    ims_ids, ime_ids = get_chatml_ids(tokenizer)

    all_ids = []
    for i, msg in enumerate(messages):
        all_ids.extend(ims_ids)
        all_ids.extend(tokenizer.encode(msg["role"] + "\n", add_special_tokens=False).ids)
        all_ids.extend(tokenizer.encode(msg["content"], add_special_tokens=False).ids)
        all_ids.extend(ime_ids)
        if i < len(messages) - 1:
            all_ids.extend(tokenizer.encode("\n", add_special_tokens=False).ids)

    if add_generation_prompt:
        all_ids.extend(tokenizer.encode("\n", add_special_tokens=False).ids)
        all_ids.extend(ims_ids)
        all_ids.extend(tokenizer.encode("assistant\n", add_special_tokens=False).ids)

    return all_ids


def load_dataset_jsonl(filepath):
    """Load a JSONL dataset."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples from {filepath}")
    return examples
