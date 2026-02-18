#!/usr/bin/env python3
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─────────────────── relation maps (subset) ────────────────────────
REL_NAMES = {
    11: "Complain officially",
    16: "Sign formal agreement",
    19: "Increase military alert status",
    20: "Impose embargo, boycott, or sanctions",
    21: "Provide economic aid",
    29: "Grant diplomatic recognition",
    45: "Reduce or break diplomatic relations",
    47: "Engage in diplomatic cooperation",
    52: "Cooperate militarily",
    53: "Mobilize or increase armed forces",
    65: "Provide humanitarian aid",
    67: "Occupy territory",
    80: "Share intelligence or information",
    88: "Assassinate",
    95: "Ease economic sanctions, boycott, embargo",
    97: "Cooperate economically",
    105: "Provide military aid",
    112: "Express intent to de-escalate military engagement",
    118: "Declare truce, ceasefire",
    127: "Express intent to cooperate on intelligence",
    132: "Halt negotiations",
    137: "Threaten with sanctions, boycott, embargo",
    143: "Increase police alert status",
    176: "Reject military cooperation",
    48: "Make optimistic comment",
}
NAME2REL = {v: k for k, v in REL_NAMES.items()}

# ─────────────────── multihop rules with op flag ───────────────────
RULES = [
    # --------------- positive cascades (add) ----------------
    {"explicit":"Sign formal agreement",    "b_to_c":"Engage in diplomatic cooperation", "implicit":["Engage in diplomatic cooperation"],                       "op":"add"},
    {"explicit":"Grant diplomatic recognition","b_to_c":"Engage in diplomatic cooperation","implicit":["Engage in diplomatic cooperation"],                     "op":"add"},
    {"explicit":"Provide economic aid",      "b_to_c":"Cooperate economically",          "implicit":["Provide economic aid"],                                   "op":"add"},
    {"explicit":"Provide military aid",      "b_to_c":"Cooperate militarily",            "implicit":["Provide military aid"],                                   "op":"add"},
    {"explicit":"Mobilize or increase armed forces", "b_to_c":"Cooperate militarily",    "implicit":["Increase military alert status"],                         "op":"add"},
    {"explicit":"Declare truce, ceasefire",  "b_to_c":"Engage in diplomatic cooperation", "implicit":["Express intent to de-escalate military engagement"],    "op":"add"},
    {"explicit":"Ease economic sanctions, boycott, embargo","b_to_c":"Cooperate economically","implicit":["Cooperate economically"],                             "op":"add"},
    {"explicit":"Sign formal agreement",     "b_to_c":"Cooperate militarily",            "implicit":["Cooperate militarily"],                                   "op":"add"},
    {"explicit":"Sign formal agreement",     "b_to_c":"Share intelligence or information","implicit":["Share intelligence or information"],                     "op":"add"},
    {"explicit":"Provide military aid",      "b_to_c":"Express intent to cooperate on intelligence","implicit":["Express intent to cooperate on intelligence"],"op":"add"},
    {"explicit":"Provide economic aid",      "b_to_c":"Provide humanitarian aid",        "implicit":["Provide humanitarian aid"],                               "op":"add"},
    {"explicit":"Occupy territory",          "b_to_c":"Engage in diplomatic cooperation", "implicit":["Complain officially"],                                   "op":"add"},
    {"explicit":"Impose embargo, boycott, or sanctions","b_to_c":"Engage in diplomatic cooperation","implicit":["Threaten with sanctions, boycott, embargo"],    "op":"add"},
    {"explicit":"Grant diplomatic recognition","b_to_c":"Engage in diplomatic cooperation","implicit":["Grant diplomatic recognition"],                         "op":"add"},
    {"explicit":"Provide humanitarian aid",  "b_to_c":"Engage in diplomatic cooperation", "implicit":["Engage in material cooperation"],                        "op":"add"},
    {"explicit":"Ease economic sanctions, boycott, embargo","b_to_c":"Engage in diplomatic cooperation","implicit":["Make optimistic comment"],                 "op":"add"},

    # --------------- negative cascades (delete) --------------------
    {"explicit":"Reduce or break diplomatic relations","b_to_c":"Engage in diplomatic cooperation","implicit":["Reduce or break diplomatic relations"],          "op":"del"},
    {"explicit":"Impose embargo, boycott, or sanctions","b_to_c":"Cooperate economically", "implicit":["Impose embargo, boycott, or sanctions"],                 "op":"del"},
    {"explicit":"Reduce or break diplomatic relations","b_to_c":"Cooperate economically", "implicit":["Impose embargo, boycott, or sanctions"],                 "op":"del"},
    {"explicit":"Mobilize or increase armed forces",   "b_to_c":"Engage in diplomatic cooperation","implicit":["Increase military alert status"],                "op":"del"},
    {"explicit":"Reduce or break diplomatic relations","b_to_c":"Engage in diplomatic cooperation","implicit":["Halt negotiations"],                            "op":"del"},
    {"explicit":"Reduce or stop military assistance",  "b_to_c":"Cooperate militarily", "implicit":["Reject military cooperation"],                             "op":"del"},
    {"explicit":"Threaten with military force",        "b_to_c":"Engage in diplomatic cooperation","implicit":["Mobilize or increase armed forces"],             "op":"del"},
]

RULES_2HOP = [
    # A signs agreement with B ;  B ↔ diplomatic coop C ;  C ↔ military coop D
    # → A and D now military-cooperate
    {"explicit":"Sign formal agreement",
     "b_to_c":"Engage in diplomatic cooperation",
     "c_to_d":"Cooperate militarily",
     "implicit":["Cooperate militarily"],
     "op":"add"},

    # A sanctions B ;  B trades with C ;  C trades with D
    # → A sanctions D as well
    {"explicit":"Impose embargo, boycott, or sanctions",
     "b_to_c":"Cooperate economically",
     "c_to_d":"Cooperate economically",
     "implicit":["Impose embargo, boycott, or sanctions"],
     "op":"add"},
]

RULES_2HOP.extend([
    # 1. Economic aid → humanitarian aid chain
    {"explicit": "Provide economic aid",
     "b_to_c": "Provide humanitarian aid",
     "c_to_d": "Provide humanitarian aid",
     "implicit": ["Provide humanitarian aid"],
     "op": "add"},

    # 2. Economic aid → humanitarian aid → economic aid
    {"explicit": "Provide economic aid",
     "b_to_c": "Provide humanitarian aid",
     "c_to_d": "Provide economic aid",
     "implicit": ["Provide economic aid"],
     "op": "add"},

    # 3. Mutual recognition chain
    {"explicit": "Grant diplomatic recognition",
     "b_to_c": "Grant diplomatic recognition",
     "c_to_d": "Grant diplomatic recognition",
     "implicit": ["Grant diplomatic recognition"],
     "op": "add"},

    # 4. Agreement → recognition → cooperation
    {"explicit": "Sign formal agreement",
     "b_to_c": "Grant diplomatic recognition",
     "c_to_d": "Engage in diplomatic cooperation",
     "implicit": ["Engage in diplomatic cooperation"],
     "op": "add"},

    # 5. Agreement → recognition → recognition
    {"explicit": "Sign formal agreement",
     "b_to_c": "Grant diplomatic recognition",
     "c_to_d": "Grant diplomatic recognition",
     "implicit": ["Grant diplomatic recognition"],
     "op": "add"},

    # 6. Military aid → military cooperation → alert
    {"explicit": "Provide military aid",
     "b_to_c": "Cooperate militarily",
     "c_to_d": "Increase military alert status",
     "implicit": ["Increase military alert status"],
     "op": "add"},

    # 7. Military aid → military cooperation → force
    {"explicit": "Provide military aid",
     "b_to_c": "Cooperate militarily",
     "c_to_d": "Use conventional military force",
     "implicit": ["Use conventional military force"],
     "op": "add"},

    # 8. Sanctions → sanctions → ease sanctions (delete)
    {"explicit": "Impose embargo, boycott, or sanctions",
     "b_to_c": "Impose embargo, boycott, or sanctions",
     "c_to_d": "Ease economic sanctions, boycott, embargo",
     "implicit": ["Ease economic sanctions, boycott, embargo"],
     "op": "del"},

    # 9. Ease sanctions → ease sanctions → impose sanctions
    {"explicit": "Ease economic sanctions, boycott, embargo",
     "b_to_c": "Ease economic sanctions, boycott, embargo",
     "c_to_d": "Impose embargo, boycott, or sanctions",
     "implicit": ["Impose embargo, boycott, or sanctions"],
     "op": "add"},

    # 10. Cease-fire → de-escalate → cooperation
    {"explicit": "Declare truce, ceasefire",
     "b_to_c": "Express intent to de-escalate military engagement",
     "c_to_d": "Engage in diplomatic cooperation",
     "implicit": ["Engage in diplomatic cooperation"],
     "op": "add"},

    # 11. Cease-fire → cease-fire → de-escalate
    {"explicit": "Declare truce, ceasefire",
     "b_to_c": "Declare truce, ceasefire",
     "c_to_d": "Express intent to de-escalate military engagement",
     "implicit": ["Express intent to de-escalate military engagement"],
     "op": "add"},

    # 12. Assassinate → assassinate → break relations (delete)
    {"explicit": "Assassinate",
     "b_to_c": "Assassinate",
     "c_to_d": "Reduce or break diplomatic relations",
     "implicit": ["Reduce or break diplomatic relations"],
     "op": "del"},

    # 13. Agreement → recognition → asylum
    {"explicit": "Sign formal agreement",
     "b_to_c": "Grant diplomatic recognition",
     "c_to_d": "Grant asylum",
     "implicit": ["Grant asylum"],
     "op": "add"},

    # 14. Asylum → asylum → humanitarian aid
    {"explicit": "Grant asylum",
     "b_to_c": "Grant asylum",
     "c_to_d": "Provide humanitarian aid",
     "implicit": ["Provide humanitarian aid"],
     "op": "add"},

    # 15. Humanitarian aid → humanitarian aid → economic aid
    {"explicit": "Provide humanitarian aid",
     "b_to_c": "Provide humanitarian aid",
     "c_to_d": "Provide economic aid",
     "implicit": ["Provide economic aid"],
     "op": "add"},

    # 16. Humanitarian aid → economic aid → cooperation
    {"explicit": "Provide humanitarian aid",
     "b_to_c": "Provide economic aid",
     "c_to_d": "Cooperate economically",
     "implicit": ["Cooperate economically"],
     "op": "add"},

    # 17. Emergency → police alert → military alert
    {"explicit": "Impose state of emergency or martial law",
     "b_to_c": "Increase police alert status",
     "c_to_d": "Increase military alert status",
     "implicit": ["Increase military alert status"],
     "op": "add"},

    # 18. Police alert → military alert → cooperation
    {"explicit": "Increase police alert status",
     "b_to_c": "Increase military alert status",
     "c_to_d": "Cooperate militarily",
     "implicit": ["Cooperate militarily"],
     "op": "add"},

    # 19. Military cooperation → military cooperation → mobilize
    {"explicit": "Cooperate militarily",
     "b_to_c": "Cooperate militarily",
     "c_to_d": "Mobilize or increase armed forces",
     "implicit": ["Mobilize or increase armed forces"],
     "op": "add"},

    # 20. Negotiation → agreement → recognition
    {"explicit": "Engage in negotiation",
     "b_to_c": "Sign formal agreement",
     "c_to_d": "Grant diplomatic recognition",
     "implicit": ["Grant diplomatic recognition"],
     "op": "add"},
])



# ─────────────────── KG loading / helpers ──────────────────────────
def load_kg(path: Path):
    kg = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            s, r, t, ds = line.rstrip("\n").split("\t")
            d = datetime.strptime(ds, "%Y-%m-%d").date()
            kg[d][s].append((r, t))
            kg[d][t].append((r, s))
    return kg

def agg_edges_up_to(kg, entity, cutoff_date):
    """Return set[(rel, nbr)] seen for `entity` on or before cutoff_date."""
    edges = set()
    for d in kg:                                # kg keys are date objects
        if d <= cutoff_date and entity in kg[d]:
            edges.update(kg[d][entity])         # list of (rel, nbr)
    return edges

def one_hop(kg, day, *ents):
    edges = set()
    for e in ents:
        for r, nbr in kg[day].get(e, []):
            edges.add((e, r, nbr))
    return edges

NUM_ROUNDS = 3      # 1-hop + one more hop over newly added edges

def apply_rules_once(current_edges, a, rel_exp, b):
    """
    current_edges : set[(h,r,t)]
    returns (added_set, removed_set) caused in ONE round.
    """
    added, removed = set(), set()

    # Build lookup (undirected) from current_edges for convenience
    out_map = defaultdict(list)
    for h, r, t in current_edges:
        out_map[h].append((r, t))

    for rule in RULES:
        if rel_exp != rule["explicit"]:
            continue
        # B → C edges (use whatever exists *now*)
        for rel_bc, c in out_map[b]:
            if rel_bc != rule["b_to_c"]:
                continue

            edge_set = {(a, r_imp, c) for r_imp in rule["implicit"]}
            if rule["op"] == "add":
                added |= edge_set - current_edges
            else:  # "del"
                removed |= {e for e in edge_set if e in current_edges}

    return added, removed


# ───────────────────────── main script ─────────────────────────────
def main():
    train_path = Path("./data/icews14/train.txt")
    out_path   = Path("./data/icews14/train_gupdate_rule_cascade.jsonl")
    kg = load_kg(train_path)

    total = added_total = removed_total = 0
    hop_add_counts = Counter()
    hop_del_counts = Counter()
    added_cnt = Counter()
    removed_cnt = Counter()

    with out_path.open("w", encoding="utf-8") as jout, train_path.open(encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Generating cascading updates"):
            a, rel_exp, b, ds = line.rstrip("\n").split("\t")
            day = datetime.strptime(ds, "%Y-%m-%d").date()

            sub_before = one_hop(kg, day, a, b)
            # Initialize
            all_added, all_removed = set(), set()
            current_edges = set(sub_before)

            for round_num in range(NUM_ROUNDS):
                round_added, round_removed = set(), set()

                # Build neighbor map from current edges
                neighbor_map = defaultdict(list)
                for h, r, t in current_edges:
                    neighbor_map[h].append((r, t))

                for rule in RULES:
                    for (h, r, t) in current_edges:
                        if r != rule["explicit"]:
                            continue
                        # t is the pivot (like entity B)
                        for rel_bc, c in agg_edges_up_to(kg, b, day):  # B → C edges
                            if rel_bc != rule["b_to_c"]:
                                continue
                            for r_imp in rule["implicit"]:
                                inferred = (h, r_imp, c)
                                if h == c:
                                    continue
                                if rule["op"] == "add":
                                    if inferred not in current_edges:
                                        round_added.add(inferred)
                                else:
                                    if inferred in current_edges:
                                        round_removed.add(inferred)


                # if not round_added and not round_removed:
                #     break

                current_edges = (current_edges | round_added) - round_removed
                all_added |= round_added
                all_removed |= round_removed

                hop_add_counts[round_num + 1] += len(round_added)
                hop_del_counts[round_num + 1] += len(round_removed)


            # Skip if nothing changed
            if not all_added and not all_removed:
                continue

            total += 1
            added_total += len(all_added)
            removed_total += len(all_removed)

            sub_after = (sub_before | all_added) - all_removed

            jout.write(json.dumps({
                "trigger_event": [a, rel_exp, b, ds],
                "edge_diff": {
                    "added": [list(e) for e in all_added],
                    "removed": [list(e) for e in all_removed]
                },
                "subgraph_before": [list(e) for e in sub_before],
                "subgraph_after": [list(e) for e in sub_after]
            }, ensure_ascii=False) + "\n")


    print("\n==== Cascading Edge-diff Statistics ====")
    print(f"Total triggers processed : {total}")
    print(f"Total edges added        : {added_total}")
    print(f"Total edges removed      : {removed_total}")
    if total:
        print(f"Avg added per trigger    : {added_total/total:.2f}")
        print(f"Avg removed per trigger  : {removed_total/total:.2f}")
    print("Hop-wise edge additions:")
    for hop in sorted(hop_add_counts):
        print(f"  Round {hop}: {hop_add_counts[hop]} added, {hop_del_counts[hop]} removed")

if __name__ == "__main__":
    main()