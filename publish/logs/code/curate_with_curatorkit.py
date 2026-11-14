import json, collections
from curatorkit import Curator, CuratorConfig
src="/mnt/user-data/uploads/lexsi-hackathon/data/codemix_v1.jsonl"
res={}
for s in ["en","hi_deva","hi_rom","pa_guru","pa_rom"]:
    rows=[json.loads(l) for l in open(src) if json.loads(l)["script"]==s]
    fn=f"in_{s}.jsonl"
    with open(fn,"w") as f:
        for r in rows: f.write(json.dumps({"instruction":r["text"],"id":r["id"],"label":r["label"]},ensure_ascii=False)+"\n")
    cfg=CuratorConfig(dataset=fn, format="auto", min_tokens=1, max_tokens=2048, dedup="minhash", minhash_threshold=0.85, clean=True, output_dir=f"out_{s}")
    r=Curator(cfg).run()
    res[s]=r
    print(s, "in",len(rows), "| result:", {k:getattr(r,k) for k in dir(r) if not k.startswith('_') and isinstance(getattr(r,k),(int,float,str))})
