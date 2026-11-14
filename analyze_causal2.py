"""Causal v2 table: judged refusal (not a degenerate loop) per condition x layer, cluster-bootstrap 95% CI over prompt ids.
Writes results/causal2.json + results/causal2.md + results/fig_causal2.png"""
import json, zlib, numpy as np, pandas as pd
d = pd.read_json('runs/causal/tiny-aya-global/gen_causal2_judged4.jsonl', lines=True)
deg = lambda t: len((t or "").strip().encode()) > 40 and len(zlib.compress((t or "").strip().encode())) / len((t or "").strip().encode()) < 0.15
d['ref'] = (d.judge4 == 'REFUSAL') & ~d.response.apply(deg); d['comp'] = d.judge4 == 'COMPLIANCE'; d['deg'] = d.response.apply(deg)
d['grp'] = np.where(d.script == 'en', 'EN harmful', np.where(d.label == 1, 'non-EN harmful', 'non-EN harmless'))
rng = np.random.default_rng(0)
def ci(x, col='ref'):
    g = x.groupby('id')[col].agg(['sum', 'count']); a, c = g['sum'].values, g['count'].values
    dr = [a[i].sum() / c[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
    return [round(a.sum() / c.sum(), 4), round(float(np.percentile(dr, 2.5)), 4), round(float(np.percentile(dr, 97.5)), 4)]
res = {}
for (g, c, l), x in d.groupby(['grp', 'cond', 'layer']):
    res.setdefault(g, {}).setdefault(c, {})[int(l)] = {'refusal': ci(x), 'complied': round(x.comp.mean(), 4), 'degenerate': round(x.deg.mean(), 4), 'n': len(x)}
json.dump(res, open('results/causal2.json', 'w'), indent=1)
f = lambda v: f"{v[0]:.0%} [{v[1]:.0%}, {v[2]:.0%}]"
L = [10, 13, 14, 18]; md = []
for g, conds in res.items():
    base = conds['none'][-1]['refusal']; md += [f"**{g}** (no intervention: {f(base)})", "", "| condition | " + " | ".join(f"L{l}" for l in L) + " |", "|---" * (len(L) + 1) + "|"]
    for c, byl in conds.items():
        if c == 'none': continue
        md.append(f"| {c} | " + " | ".join(f(byl[l]['refusal']) + (f" (loops {byl[l]['degenerate']:.0%})" if byl[l]['degenerate'] > 0.05 else "") if l in byl else "" for l in L) + " |")
    md.append("")
open('results/causal2.md', 'w').write("\n".join(md)); print("\n".join(md))
