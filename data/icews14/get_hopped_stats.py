import json
import numpy as np

cascade_file = "./data/icews14/train_gupdate_cascade.json"

with open(cascade_file, "r") as f:
    cascades = json.load(f)
print(len(cascades))
# cascades is expected to be a dict: {idx: [hop0, hop1, hop2, ...]}
hop1_counts = []
hop2_counts = []
total_connections = 0

for idx, hops in cascades.items():
    # hop0 = [seed record]
    if len(hops) > 1:
        hop1 = hops[1]
        hop1_counts.append(len(hop1))
        total_connections += len(hop1)
    else:
        hop1_counts.append(0)

    if len(hops) > 2:
        hop2 = hops[2]  # list of lists
        hop2_flat = sum(len(h) for h in hop2)
        hop2_counts.append(hop2_flat)
        total_connections += hop2_flat
    else:
        hop2_counts.append(0)

stats = {
    "total_trigger_events": len(cascades),
    "total_connections": total_connections,
    "avg_hop1": float(np.mean(hop1_counts)),
    "max_hop1": int(np.max(hop1_counts)),
    "min_hop1": int(np.min(hop1_counts)),
    "avg_hop2": float(np.mean(hop2_counts)),
    "max_hop2": int(np.max(hop2_counts)),
    "min_hop2": int(np.min(hop2_counts)),
}

print(stats)
print(json.dumps(stats, indent=2))

with open(f"./data/icews14/train_gupdate_cascade_stats.json", "w") as f:
    json.dump(stats, f)