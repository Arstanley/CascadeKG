from torch.utils.data import Dataset
import json
import torch
from torch_geometric.data import Data
import pickle as pkl
import numpy as np

class CascadeDataset(Dataset):
    def __init__(self, json_file, embed_file, entity2id, relation2id, num_hops=2):
        with open(json_file, "r") as f:
            self.cascades = json.load(f)
        
        with open(embed_file, "rb") as f:
            # Load embeddings and convert to Torch Tensor immediately for slicing
            raw_emb = pkl.load(f)
            if isinstance(raw_emb, np.ndarray):
                self.node_emb = torch.from_numpy(raw_emb).float()
            else:
                self.node_emb = torch.tensor(raw_emb).float()

        self.num_hops = num_hops
        self.entity2id = entity2id
        self.relation2id = relation2id
        
        # Pre-build full hierarchical structures for each root
        self.records = []
        for root, hops in self.cascades.items():
            # We start processing from hop 0
            root_struct = self._build_recursive_structure(hops, current_hop=0)
            self.records.append(root_struct)

    def _get_triplets(self, raw_edges):
        """Helper to convert raw edge lists to global ID triplets"""
        return [[self.entity2id[s], self.relation2id[r], self.entity2id[t]] for s, r, t in raw_edges]

    def _create_data_with_mapping(self, triplets, global_to_local, x_subset, subset_ids):
        """
        Creates a Data object using a PRE-EXISTING global_to_local mapping.
        This allows multiple graphs (before, after, children) to share the same node indices.
        """
        if not triplets:
            return Data(
                x=x_subset, 
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                n_id=subset_ids
            )

        edge_index = []
        edge_type = []
        
        for s, r, o in triplets:
            # If a node is missing from the map (shouldn't happen if map is built correctly), skip
            if s in global_to_local and o in global_to_local:
                edge_index.append([global_to_local[s], global_to_local[o]])
                edge_type.append(r)
        
        if not edge_index:
             return Data(
                x=x_subset, 
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                n_id=subset_ids
            )

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        return Data(x=x_subset, edge_index=edge_index, edge_type=edge_type, n_id=subset_ids)

    def _build_recursive_structure(self, hops, current_hop=0):
        if current_hop >= len(hops) or current_hop > self.num_hops:
            return []

        current_records = hops[current_hop]
        result = []

        for record in current_records:
            # Handle nested hop2 (list of lists)
            if current_hop == 2 and isinstance(record, list) and isinstance(record[0], list):
                for subrecord in record:
                    result.extend(self._parse_record(subrecord, hops, current_hop))
            else:
                result.extend(self._parse_record(record, hops, current_hop))

        return result

    def _parse_record(self, record, hops, current_hop):
        trigger_event, subgraph_before, subgraph_after, paragraph = record

        # --- 1. Identify Children (Next Hop) Records ---
        raw_child_records = []
        if current_hop + 1 < len(hops) and current_hop + 1 <= self.num_hops:
            next_hop = hops[current_hop + 1]
            if current_hop + 1 == 2 and isinstance(next_hop[0], list):
                raw_child_records = [sub for sublist in next_hop for sub in sublist]
            else:
                raw_child_records = next_hop

        # --- 2. Gather ALL Global IDs for this Context (Current + Immediate Children) ---
        # We need the UNION of nodes so that 'subgraph_before' knows about 'child_subgraph_after' nodes.
        
        # A. Current Level Triplets
        sb_trips = self._get_triplets(subgraph_before)
        sa_trips = self._get_triplets(subgraph_after)
        trig_trip = [self.entity2id[trigger_event[0]], self.relation2id[trigger_event[1]], self.entity2id[trigger_event[2]]]
        si_trips = sb_trips + [trig_trip]

        # B. Child Level Triplets (Lookahead)
        child_triplets_collection = []
        for child in raw_child_records:
            c_trig, c_bef, c_aft, _ = child
            c_sb = self._get_triplets(c_bef)
            c_sa = self._get_triplets(c_aft)
            c_trig_t = [self.entity2id[c_trig[0]], self.relation2id[c_trig[1]], self.entity2id[c_trig[2]]]
            child_triplets_collection.append({
                "before": c_sb, 
                "after": c_sa, 
                "inter": c_sb + [c_trig_t],
                "trigger": c_trig,
                "para": child[3]
            })

        # C. Build Unique Node Set (The Super-Set)
        all_nodes = set()
        
        # Add current nodes
        for t_list in [sb_trips, sa_trips, si_trips]:
            for s, r, o in t_list:
                all_nodes.add(s); all_nodes.add(o)
        
        # Add child nodes
        for c_dict in child_triplets_collection:
            for t_list in [c_dict["before"], c_dict["after"], c_dict["inter"]]:
                for s, r, o in t_list:
                    all_nodes.add(s); all_nodes.add(o)

        unique_nodes = sorted(list(all_nodes))
        
        # --- 3. Create Unified Embedding and Mapping ---
        subset_ids = torch.tensor(unique_nodes, dtype=torch.long)
        x_unified = self.node_emb[subset_ids] # This is the Embedding Matrix for everyone
        
        global_to_local = {gid: i for i, gid in enumerate(unique_nodes)}

        # --- 4. Build Data Objects for Current Level ---
        d_bef = self._create_data_with_mapping(sb_trips, global_to_local, x_unified, subset_ids)
        d_int = self._create_data_with_mapping(si_trips, global_to_local, x_unified, subset_ids)
        d_aft = self._create_data_with_mapping(sa_trips, global_to_local, x_unified, subset_ids)

        # --- 5. Build Data Objects for Children (aligned to Parent) ---
        newly_added_edges = []
        for i, c_dict in enumerate(child_triplets_collection):
            c_d_bef = self._create_data_with_mapping(c_dict["before"], global_to_local, x_unified, subset_ids)
            c_d_int = self._create_data_with_mapping(c_dict["inter"], global_to_local, x_unified, subset_ids)
            c_d_aft = self._create_data_with_mapping(c_dict["after"], global_to_local, x_unified, subset_ids)
            
            child_entry = {
                "trigger_event": c_dict["trigger"],
                "subgraph_before": c_d_bef,
                "subgraph_intermediate": c_d_int,
                "subgraph_after": c_d_aft,
                "paragraph": c_dict["para"],
                "newly_added_edges": [] # Calibration only looks 1 hop deep, so we leave this empty here
            }
            newly_added_edges.append(child_entry)

        # --- 6. Recurse (Optional / For Training Deep Cascades) ---
        # Note: The recursion creates *independent* records for the children (acting as roots).
        # We don't merge them into the current record's 'newly_added_edges' because 
        # those need the parent's mapping, while recursion creates new mappings.
        # The 'newly_added_edges' we built in Step 5 is sufficient for the Calibrator.
        
        # If you need to train on children as roots, we accumulate them in the return list
        recursive_results = []
        
        # Add Current Record
        recursive_results.append({
            "trigger_event": trigger_event,
            "subgraph_before": d_bef,
            "subgraph_intermediate": d_int,
            "subgraph_after": d_aft,
            "paragraph": paragraph,
            "newly_added_edges": newly_added_edges
        })

        # Add Recursive Children (as independent roots)
        # We re-run _parse_record on them so they generate their own minimal mappings
        # This keeps the dataset versatile.
        if current_hop + 1 < len(hops) and current_hop + 1 <= self.num_hops:
             # We already extracted raw_child_records in Step 1
             for child_raw in raw_child_records:
                 recursive_results.extend(self._parse_record(child_raw, hops, current_hop + 1))

        return recursive_results

    def __len__(self):
        return len(self.records) 

    def __getitem__(self, idx):
        return self.records[idx]