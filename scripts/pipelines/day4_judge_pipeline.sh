#!/bin/bash
# Day 4: wait for generation jobs, then judge their outputs (Aya Expanse 8B, 4-way), in parallel groups.
cd ~/lexsi-hackathon; export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH; export HF_TOKEN=$(cat ~/.cache/huggingface/token)
R=runs/steer/tiny-aya-global
judge() { cd /tmp; hf jobs uv run --detach --flavor a10g-small --timeout 90m --secrets HF_TOKEN ~/lexsi-hackathon/scripts/jobs/job_judge4.py sunit12/aya-codemix-work "$@" 2>&1 | grep -oE "sunit12/[0-9a-f]{24}" | head -1; cd ~/lexsi-hackathon; }
waitj() { for J in "$@"; do hf jobs wait sunit12/$J >/dev/null 2>&1 || true; done; }
( waitj 6af8c97729dd05ff0367c87e 6af8c97c29dd05ff0367c88d 6af8ca9e29dd05ff0367c9d9
  echo "judge unseen+attacks: $(judge $R/gen_control_unseen_langs_v1.jsonl $R/gen_g3_14_unseen_langs_v1.jsonl $R/gen_g31_14_unseen_langs_v1.jsonl $R/gen_control_attacks_v1.jsonl $R/gen_g3_14_attacks_v1.jsonl) $(date)" ) &
( waitj 6af8c99e29dd05ff0367c8e1; echo "judge teacher: $(judge runs/distill/tiny-aya-global/teacher.jsonl) $(date)" ) &
( waitj 6af8c77023985fb9cd82875c; echo "judge causal: $(judge runs/causal/tiny-aya-global/gen_causal.jsonl) $(date)" ) &
( waitj 6af8cab229dd05ff0367ca02 6af8cab629dd05ff0367ca06
  echo "judge filler+everyday: $(judge $R/gen_control_filler_test.jsonl $R/gen_g3_14_filler_test.jsonl $R/gen_control_everyday_test.jsonl $R/gen_g3_14_everyday_test.jsonl) $(date)" ) &
wait; echo "ALL JUDGES LAUNCHED $(date)"
