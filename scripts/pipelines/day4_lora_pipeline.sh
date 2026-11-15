#!/bin/bash
# Day 4: for each LoRA training job, when done: judge its eval outputs, run attack eval + AuditKIT MMLU, judge attacks.
cd ~/lexsi-hackathon; export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH; export HF_TOKEN=$(cat ~/.cache/huggingface/token)
L=runs/lora/tiny-aya-global; W=sunit12/aya-codemix-work; M=CohereLabs/tiny-aya-global
launch() { cd /tmp; hf jobs uv run --detach --flavor "$1" --timeout 90m --secrets HF_TOKEN "${@:2}" 2>&1 | grep -oE "sunit12/[0-9a-f]{24}" | head -1; cd ~/lexsi-hackathon; }
waitj() { for J in "$@"; do hf jobs wait sunit12/$J >/dev/null 2>&1 || true; done; }
one() {  # $1 train job id, $2 run name, $3 adapter repo
  waitj $1; echo "[$2] training+eval job done $(date)"
  echo "[$2] judge A: $(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_judge4.py $W $L/gen_$2_heldout.jsonl $L/gen_$2_newcats_test.jsonl)"
  echo "[$2] judge B: $(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_judge4.py $W $L/gen_$2_unseen_langs_v1.jsonl $L/gen_$2_everyday_test.jsonl)"
  echo "[$2] judge C: $(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_judge4.py $W $L/gen_$2_filler_test.jsonl)"
  echo "[$2] mmlu: $(launch a10g-small -e GATE_FILE=gate_v3.npz -e ADAPTER=$3 ~/lexsi-hackathon/scripts/jobs/job_capability.py $W $M 400)"
  A=$(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_eval_adapter.py $W $M $3 $2 attacks_v1.jsonl); echo "[$2] attacks gen: $A"
  waitj $A; echo "[$2] judge attacks: $(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_judge4.py $W $L/gen_$2_attacks_v1.jsonl)"
}
one 6af8d3cf23985fb9cd829508 lora_v1 sunit12/tiny-aya-global-codemix-refusal-lora &
one 6af8dbe523985fb9cd829deb lora_v2 sunit12/tiny-aya-global-codemix-refusal-lora-v2 &
one 6af8dbe829dd05ff0367df56 lora_v2c sunit12/tiny-aya-global-codemix-refusal-lora-v2-circuit &
# v0 leftovers: everyday judge + attacks judge
( waitj 6af8d32129dd05ff0367d444; echo "[lora_v0] judge everyday: $(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_judge4.py $W $L/gen_lora_v0_everyday_test.jsonl)" ) &
( waitj 6af8dc2223985fb9cd829e1f; echo "[lora_v0] judge attacks: $(launch a10g-small ~/lexsi-hackathon/scripts/jobs/job_judge4.py $W $L/gen_lora_v0_attacks_v1.jsonl)" ) &
wait; echo "LORA PIPELINE DONE $(date)"
