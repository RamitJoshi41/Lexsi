### Baseline

| model           | script   | refuse        | comply   | partial   | non-answer   | harmless refused   | harmless non-answer   |
|:----------------|:---------|:--------------|:---------|:----------|:-------------|:-------------------|:----------------------|
| tiny-aya-global | en       | 83% [75%–91%] | 12%      | 5%        | 0%           | 11%                | 1%                    |
| tiny-aya-global | hi_deva  | 66% [56%–76%] | 33%      | 1%        | 0%           | 6%                 | 0%                    |
| tiny-aya-global | hi_rom   | 30% [19%–40%] | 43%      | 19%       | 8%           | 10%                | 9%                    |
| tiny-aya-global | pa_guru  | 62% [52%–73%] | 36%      | 1%        | 0%           | 17%                | 0%                    |
| tiny-aya-global | pa_rom   | 22% [12%–31%] | 28%      | 15%       | 34%          | 14%                | 19%                   |
| tiny-aya-fire   | en       | 89% [82%–94%] | 11%      | 0%        | 0%           | 10%                | 0%                    |
| tiny-aya-fire   | hi_deva  | 74% [65%–83%] | 24%      | 1%        | 0%           | 5%                 | 0%                    |
| tiny-aya-fire   | hi_rom   | 33% [23%–42%] | 39%      | 11%       | 17%          | 15%                | 16%                   |
| tiny-aya-fire   | pa_guru  | 75% [66%–84%] | 25%      | 0%        | 0%           | 20%                | 0%                    |
| tiny-aya-fire   | pa_rom   | 31% [22%–41%] | 23%      | 6%        | 41%          | 13%                | 36%                   |

### Held-out

| setting                           | refuse harmful · en   | refuse harmful · hi_deva   | refuse harmful · hi_rom   | refuse harmful · pa_guru   | refuse harmful · pa_rom   | avg refuse harmful (non-EN)   | avg refuse harmless   | avg non-answer harmless   |
|:----------------------------------|:----------------------|:---------------------------|:--------------------------|:---------------------------|:--------------------------|:------------------------------|:----------------------|:--------------------------|
| No intervention (control)         | 74%                   | 59%                        | 30%                       | 52%                        | 20%                       | 40%                           | 13%                   | 6%                        |
| Ours: gated (v2) steer L14        | 96%                   | 98%                        | 80%                       | 91%                        | 96%                       | 91%                           | 15%                   | 6%                        |
| Ours: gated (v2) steer L18        | 96%                   | 98%                        | 89%                       | 87%                        | 85%                       | 90%                           | 15%                   | 6%                        |
| Ours: gate v1 (XSTest-only) L14   | 96%                   | 98%                        | 80%                       | 89%                        | 91%                       | 90%                           | 13%                   | 6%                        |
| SafeTune CAST (English calib.)    | 83%                   | 70%                        | 35%                       | 28%                        | 22%                       | 39%                           | 14%                   | 7%                        |
| SafeTune CAST (all-script calib.) | 100%                  | 96%                        | 87%                       | 89%                        | 78%                       | 88%                           | 17%                   | 6%                        |
| Always-on steer L14               | 100%                  | 100%                       | 85%                       | 91%                        | 96%                       | 93%                           | 88%                   | 0%                        |
| Always-on steer L18               | 100%                  | 100%                       | 93%                       | 89%                        | 85%                       | 92%                           | 93%                   | 1%                        |
