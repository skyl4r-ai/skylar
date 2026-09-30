# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
DPO entry point — INTENTIONALLY NOT IMPLEMENTED (use ORPO/SimPO instead).

DPO keeps a frozen *reference* copy of the policy in memory (~2× the weights). For the
small, from-scratch models this framework targets, the reference-FREE methods are lighter
and competitive, so preference optimization lives in `bin.preference.py`:

    python training/bin.preference.py --base_model <sft_ckpt> \\
        --data <preference.jsonl> --loss orpo   # or --loss simpo

This file is kept as an explicit signpost, not a missing feature.
"""

import sys

if __name__ == "__main__":
    sys.exit("bin.dpo.py is not implemented — use bin.preference.py (ORPO/SimPO). See the docstring.")
