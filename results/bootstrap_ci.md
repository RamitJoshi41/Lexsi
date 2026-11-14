### heldout (cluster bootstrap over 94 prompt ids, 10000 draws)

| setting | usable refusal non-EN | judge refusal non-EN | harmless refused |
|---|---|---|---|
| control | 38% [31%, 46%] | 40% [32%, 49%] | 13% [8%, 20%] |
| cast_en | 7% [3%, 10%] | 39% [31%, 46%] | 14% [8%, 20%] |
| cast_all | 40% [32%, 48%] | 88% [80%, 94%] | 17% [11%, 24%] |
| gate_v2 | 81% [75%, 86%] | 91% [86%, 95%] | 15% [9%, 21%] |
| gate_v3 | 76% [69%, 83%] | 86% [80%, 92%] | 15% [9%, 21%] |
| always_on | 82% [77%, 87%] | 93% [89%, 96%] | 88% [82%, 92%] |

| difference | point | 95% CI |
|---|---|---|
| gate_v3 - cast_all (usable, non-EN) | +36 pts | [+25, +46] |
| gate_v3 - control (usable, non-EN) | +38 pts | [+29, +47] |
| gate_v2 - cast_all (usable, non-EN) | +41 pts | [+31, +51] |
| cast_all - control (usable, non-EN) | +2 pts | [-10, +14] |
| baseline gap EN - hi_rom (judge refusal) | +43 pts | [+29, +58] |
| baseline gap EN - pa_rom (judge refusal) | +54 pts | [+37, +71] |

### newcats_test (cluster bootstrap over 40 prompt ids, 10000 draws)

| setting | usable refusal non-EN | judge refusal non-EN | harmless refused |
|---|---|---|---|
| control | 46% [32%, 60%] | 49% [34%, 64%] | 0% [0%, 0%] |
| cast_all | 52% [40%, 65%] | 84% [70%, 95%] | 1% [0%, 3%] |
| gate_v2 | 55% [42%, 68%] | 60% [45%, 74%] | 8% [2%, 16%] |
| gate_v3 | 75% [63%, 86%] | 82% [71%, 93%] | 4% [1%, 8%] |

| difference | point | 95% CI |
|---|---|---|
| gate_v3 - cast_all (usable, non-EN) | +22 pts | [+12, +33] |
| gate_v3 - control (usable, non-EN) | +29 pts | [+14, +44] |
| gate_v2 - cast_all (usable, non-EN) | +2 pts | [-8, +12] |
| cast_all - control (usable, non-EN) | +6 pts | [-7, +19] |
| baseline gap EN - hi_rom (judge refusal) | +55 pts | [+33, +77] |
| baseline gap EN - pa_rom (judge refusal) | +60 pts | [+38, +82] |
