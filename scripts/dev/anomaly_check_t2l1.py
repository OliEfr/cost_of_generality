"""Why do clean T2_L1 n50/n100 exceed their published (inflated!) numbers?

Compare per-batch SR and stage rates: clean re-run vs original published run, per cell.
If the clean run is uniformly better including batch 0 (identical seeds/initial states),
the difference is run-level, not carryover-related.
"""
import json
from pathlib import Path

R = Path("results")
for n in (400, 200, 100, 50):
    new = json.load(open(R / f"eval_T2_L1_n{n}_080000_fixed.json"))
    old = json.load(open(R / f"eval_T2_L1_n{n}_080000.json"))
    nb = [sum(o["success"] for o in new["outcomes"] if o["batch"] == b) / 20 for b in range(5)]
    ob = [sum(o["success"] for o in old["outcomes"] if o["batch"] == b) / 20 for b in range(5)]
    st = new.get("stages", {})
    agree = sum(
        a["success"] == b["success"]
        for a, b in zip(sorted(new["outcomes"], key=lambda o: (o["batch"], o["env"])),
                        sorted(old["outcomes"], key=lambda o: (o["batch"], o["env"])))
    ) / 100
    print(f"n{n}: clean SR={new['success_rate']:.2f} per-batch={nb}")
    print(f"      old   SR={old['success_rate']:.2f} per-batch={ob}  episode-agreement={agree:.2f}")
    print(f"      clean stages: opened={st.get('drawer_opened')} lifted={st.get('object_lifted')} "
          f"over={st.get('object_over_drawer')}")
    print(f"      old ckpt: {old['checkpoint'][-70:]}")
    print(f"      new ckpt: {new['checkpoint'][-70:]}")
