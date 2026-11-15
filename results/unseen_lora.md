### unseen_lora

| system | bn_rom usable↑ / harmless refused↓ | mr_rom usable↑ / harmless refused↓ | ta_rom usable↑ / harmless refused↓ | ur_rom usable↑ / harmless refused↓ | ALL usable | ALL harmless refused | Δusable vs control | Δharmless-refused vs control |
|---|---|---|---|---|---|---|---|---|
| control | 13% / 17% | 13% / 13% | 0% / 23% | 33% / 7% | 15% | 15% | — | — |
| guard_v3 | 40% / 40% | 57% / 17% | 50% / 27% | 83% / 10% | 57% | 23% | +42% [+32%, +53%] | +8% [+4%, +12%] |
| lora_v1 | 83% / 30% | 53% / 33% | 67% / 13% | 87% / 20% | 72% | 24% | +57% [+46%, +69%] | +9% [+0%, +19%] |
| lora_v2 | 73% / 17% | 67% / 20% | 77% / 27% | 90% / 13% | 77% | 19% | +62% [+52%, +71%] | +4% [-5%, +13%] |
| lora_v2c | 87% / 20% | 63% / 17% | 80% / 33% | 83% / 13% | 78% | 21% | +63% [+52%, +74%] | +6% [-3%, +14%] |