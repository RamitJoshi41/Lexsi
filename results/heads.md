Heads: REF=A14.13,A16.2,A20.4; PA=A31.11,A25.3,A29.13; random=[[14, 12], [16, 10], [20, 8]]

| group          | cond                | refusal        | complied   | degenerate   |   n |
|:---------------|:--------------------|:---------------|:-----------|:-------------|----:|
| en_harmful     | ablate_pa_heads     | 85% [74%, 93%] | 11%        | 0%           |  46 |
| en_harmful     | ablate_random_heads | 91% [83%, 98%] | 9%         | 0%           |  46 |
| en_harmful     | ablate_ref_heads    | 83% [72%, 91%] | 11%        | 0%           |  46 |
| en_harmful     | none                | 89% [80%, 98%] | 11%        | 0%           |  46 |
| nonEN_harmful  | amp2_ref_heads      | 55% [47%, 62%] | 23%        | 0%           | 184 |
| nonEN_harmful  | amp4_random_heads   | 49% [41%, 57%] | 23%        | 0%           | 184 |
| nonEN_harmful  | amp4_ref_heads      | 53% [45%, 62%] | 20%        | 0%           | 184 |
| nonEN_harmful  | none                | 51% [43%, 60%] | 28%        | 0%           | 184 |
| nonEN_harmless | amp4_ref_heads      | 18% [11%, 24%] | 42%        | 0%           | 192 |
| nonEN_harmless | none                | 18% [11%, 26%] | 57%        | 0%           | 192 |