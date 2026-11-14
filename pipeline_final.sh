#!/bin/bash
# waits for the gate-v3 steering/CAST jobs, judges them, runs response LID and the final comparison
cd ~/lexsi-hackathon; export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH; export HF_TOKEN=$(cat ~/.cache/huggingface/token)
R=runs/steer/tiny-aya-global
for J in 6af87a3929dd05ff03676bef 6af87a3e29dd05ff03676bf5 6af87a4223985fb9cd823718 6af87a5d23985fb9cd82372e; do hf jobs wait sunit12/$J >/dev/null 2>&1 || true; done
echo "steer jobs done $(date)"
cd /tmp; JID=$(hf jobs uv run --detach --flavor a10g-small --timeout 45m --secrets HF_TOKEN ~/lexsi-hackathon/job_judge4.py sunit12/aya-codemix-work \
  $R/gen_g3_14.jsonl $R/gen_control_newcats_test.jsonl $R/gen_g3_14_newcats_test.jsonl $R/gen_g2_14_newcats_test.jsonl $R/gen_safetune_cast_all_a1_newcats_test.jsonl 2>&1 | grep -E "^\s+id:" | awk '{print $2}')
echo "judge job $JID"; hf jobs wait sunit12/$JID >/dev/null 2>&1 || true; echo "judge done $(date)"
cd ~/lexsi-hackathon && hf download sunit12/aya-codemix-work --repo-type dataset --include "$R/*judged4.jsonl" --include "runs/capability_*" --local-dir . >/dev/null 2>&1
uv run -q response_lid.py $R/gen_g3_14_judged4.jsonl $R/gen_control_newcats_test_judged4.jsonl $R/gen_g3_14_newcats_test_judged4.jsonl $R/gen_g2_14_newcats_test_judged4.jsonl $R/gen_safetune_cast_all_a1_newcats_test_judged4.jsonl $R/gen_always14_a1_judged4.jsonl
uv run -q compare_final.py > results/final_v2.txt 2>&1
echo "ALL DONE $(date)"
