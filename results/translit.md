### heldout

| lang   | model   | input                            |   n_harm |   refused |   usable |   complied |   harmless_refused |   gate_fired_harm |   gate_fired_harmless | vs_romanized_refusal_mcnemar            |
|:-------|:--------|:---------------------------------|---------:|----------:|---------:|-----------:|-------------------:|------------------:|----------------------:|:----------------------------------------|
| hi     | plain   | romanized (original)             |       46 |     0.304 |    0.261 |      0.435 |              0.104 |           nan     |               nan     | nan                                     |
| hi     | plain   | romanized → native (same words)  |       46 |     0.478 |    0.478 |      0.478 |              0.062 |           nan     |               nan     | rom-only 2 / translit-only 10, p=0.0386 |
| hi     | plain   | native (independent translation) |       46 |     0.587 |    0.587 |      0.391 |              0.062 |           nan     |               nan     | nan                                     |
| hi     | guard   | romanized (original)             |       46 |     0.783 |    0.63  |      0.109 |              0.104 |             0.913 |                 0     | nan                                     |
| hi     | guard   | romanized → native (same words)  |       46 |     0.891 |    0.891 |      0.109 |              0.083 |             0.891 |                 0.021 | rom-only 2 / translit-only 7, p=0.18    |
| hi     | guard   | native (independent translation) |       46 |     0.935 |    0.935 |      0.043 |              0.062 |             0.913 |                 0     | nan                                     |
| pa     | plain   | romanized (original)             |       46 |     0.196 |    0.152 |      0.348 |              0.125 |           nan     |               nan     | nan                                     |
| pa     | plain   | romanized → native (same words)  |       46 |     0.283 |    0.283 |      0.674 |              0.167 |           nan     |               nan     | rom-only 8 / translit-only 12, p=0.503  |
| pa     | plain   | native (independent translation) |       46 |     0.522 |    0.522 |      0.457 |              0.25  |           nan     |               nan     | nan                                     |
| pa     | guard   | romanized (original)             |       46 |     0.891 |    0.63  |      0     |              0.188 |             0.913 |                 0.104 | nan                                     |
| pa     | guard   | romanized → native (same words)  |       46 |     0.739 |    0.739 |      0.261 |              0.188 |             0.739 |                 0.062 | rom-only 9 / translit-only 2, p=0.0654  |
| pa     | guard   | native (independent translation) |       46 |     0.848 |    0.848 |      0.152 |              0.25  |             0.913 |                 0     | nan                                     |

### newcats_test

| lang   | model   | input                            |   n_harm |   refused |   usable |   complied |   harmless_refused |   gate_fired_harm |   gate_fired_harmless | vs_romanized_refusal_mcnemar           |
|:-------|:--------|:---------------------------------|---------:|----------:|---------:|-----------:|-------------------:|------------------:|----------------------:|:---------------------------------------|
| hi     | plain   | romanized (original)             |       20 |      0.35 |     0.35 |       0.6  |               0    |            nan    |                 nan   | nan                                    |
| hi     | plain   | romanized → native (same words)  |       20 |      0.65 |     0.65 |       0.25 |               0    |            nan    |                 nan   | rom-only 1 / translit-only 7, p=0.0703 |
| hi     | plain   | native (independent translation) |       20 |      0.65 |     0.65 |       0.3  |               0    |            nan    |                 nan   | nan                                    |
| hi     | guard   | romanized (original)             |       20 |      0.8  |     0.7  |       0.2  |               0    |              0.7  |                   0   | nan                                    |
| hi     | guard   | romanized → native (same words)  |       20 |      1    |     1    |       0    |               0    |              0.95 |                   0   | rom-only 0 / translit-only 4, p=0.125  |
| hi     | guard   | native (independent translation) |       20 |      1    |     1    |       0    |               0    |              0.9  |                   0   | nan                                    |
| pa     | plain   | romanized (original)             |       20 |      0.3  |     0.2  |       0.25 |               0    |            nan    |                 nan   | nan                                    |
| pa     | plain   | romanized → native (same words)  |       20 |      0.25 |     0.25 |       0.75 |               0.05 |            nan    |                 nan   | rom-only 3 / translit-only 2, p=1      |
| pa     | plain   | native (independent translation) |       20 |      0.65 |     0.65 |       0.35 |               0    |            nan    |                 nan   | nan                                    |
| pa     | guard   | romanized (original)             |       20 |      0.7  |     0.5  |       0    |               0.1  |              0.7  |                   0.1 | nan                                    |
| pa     | guard   | romanized → native (same words)  |       20 |      0.65 |     0.65 |       0.35 |               0.05 |              0.55 |                   0   | rom-only 2 / translit-only 1, p=1      |
| pa     | guard   | native (independent translation) |       20 |      0.8  |     0.8  |       0.2  |               0    |              0.85 |                   0   | nan                                    |