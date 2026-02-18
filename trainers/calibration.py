# Now we load the multi-hop cascade data
import numpy as np
from itertools import product
import random
import math
from scipy.stats import binom
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
from utils.utils import * 

class Calibrator:
    def __init__(
        self,
        cal_dataset,
        model,
        device,
        sentence_encoder,
        model_variant,
        num_relations,
        id2ent,
        id2rel,
        grid_step=0.1,   # τ: acceptable F1 cutoff
        error_rate=0.10,     # ε: allowed failure rate
        delta=0.0005,           # δ: family-wise error rate
        split_ratio = 0.3, # Split ratio for optimization and calibration
    ):
        self.cal_dataset = cal_dataset
        self.grid_step = grid_step
        self.error_rate = error_rate
        self.delta = delta
        self.split_ratio = split_ratio
        self.model = model
        self.model_variant = model_variant
        self.sentence_encoder = sentence_encoder
        self.num_relations = num_relations    
        self.id2ent = id2ent

        self.ent2id = {v: k for k, v in id2ent.items()}
        self.id2rel = id2rel

        self.rel2id = {v: k for k, v in id2rel.items()}
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def _forward_cascade_prediction(
        self,
        newly_added_edges,
        subgraph_before,
        subgraph_intermediate,
        subgraph_after,
        trigger_event,
        paragraph,
        alpha1,
        alpha2,
        id2ent,
        id2rel,
        use_percentage=False
    ):
        # print(paragraph)
        """
        Perform 2-hop recursive cascade prediction and compute metrics (Optimized).
        Uses full Cartesian product for edge generation but vectorizes relation decoding.
        """
        self.model.eval()
        n_list = []
        n_total = 0
        fail_number = 0

        tp, fp, fn = 0, 0, 0

        # ==========================================
        # HOP 1: Prediction on Subgraph After
        # ==========================================

        # 1. Encode Text (Unchanged)
        if self.model_variant == 'text_conditioned':
            z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
            if z_text.dim() == 2 and z_text.size(0) == 1: z_text = z_text.squeeze(0)
        elif self.model_variant == 'rgat':
            z_text = self.sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(self.device)
            if z_text.dim() == 2 and z_text.size(0) == 1: z_text = z_text.squeeze(0)
        else:
            z_text = None

        # 2. Encode Graph (Unchanged)
        if self.model_variant in ['distmult', 'rgcn_baseline', 'transe', 'hitter', 'simkgc', 'kgt5']:
            z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type)
        else: # Default for rgat and others
            z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type, z_text)

        # 3. Generate Candidates (Cartesian Product of nodes in output - Unchanged)
        nodes_after = set(subgraph_after.edge_index.unique().tolist())
        for added in newly_added_edges:
            nodes_after.update(added["subgraph_after"].edge_index.unique().tolist())
        nodes_after = torch.tensor(sorted(list(nodes_after)), dtype=torch.long, device=self.device)
        
        head_mesh, tail_mesh = torch.meshgrid(nodes_after, nodes_after, indexing='ij')
        edge_index_candidates = torch.stack([head_mesh.flatten(), tail_mesh.flatten()], dim=0)
        mask = edge_index_candidates[0] != edge_index_candidates[1] # Remove self-loops
        edge_index_candidates = edge_index_candidates[:, mask]
        
        num_candidates_hop1 = edge_index_candidates.size(1)

        # 4. Decode & Predict (MEMORY OPTIMIZED CHUNKING)
        all_pred_edges_hop1, all_pred_types_hop1 = [], []
        num_candidates_hop1 = edge_index_candidates.size(1)

        # Containers for percentage-based thresholding
        all_probs_list = []
        all_edges_list = []
        all_types_list = []

        # Iterate over relations (R loops), but only keep high-scoring candidates
        for rel_id in range(self.num_relations):
            rel_tensor = torch.full((num_candidates_hop1,), rel_id, dtype=torch.long, device=self.device)
            
            # Calculate scores for this relation (O(N^2) memory spike)
            scores = self.model.decode(z1, edge_index_candidates, rel_tensor) 

            # --- IMMEDIATE FILTERING TO REDUCE MEMORY ---
            
            # 1. Calculate probabilities and normalize (on the O(N^2) tensor)
            probs = torch.sigmoid(scores)
            
            # To avoid the huge global min/max calculation, you can use a fixed global norm
            # or skip normalization if it's not strictly required for thresholding.
            # We'll use the local norm for correctness as in your original code:
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
            
            if use_percentage:
                all_probs_list.append(probs)
                all_edges_list.append(edge_index_candidates)
                all_types_list.append(rel_tensor)
            else:
                # 2. Apply Threshold Alpha 1 (Saves only the relevant predictions)
                pred_mask = probs >= alpha1
                
                # Store only the filtered (much smaller) results
                pred_edges = edge_index_candidates[:, pred_mask]
                pred_types = rel_tensor[pred_mask] # Reuse rel_tensor for types
                
                all_pred_edges_hop1.append(pred_edges)
                all_pred_types_hop1.append(pred_types)
            
            # Explicitly clear temporary tensors to help memory management
            del scores, probs, rel_tensor
            if not use_percentage:
                del pred_mask, pred_edges, pred_types
            torch.cuda.empty_cache()

        if use_percentage:
            # Aggregate all scores to determine threshold
            if len(all_probs_list) > 0:
                flat_probs = torch.cat(all_probs_list)
                flat_edges = torch.cat(all_edges_list, dim=1)
                flat_types = torch.cat(all_types_list, dim=0)

                # alpha1 is percentage to KEEP (e.g. 0.1 = top 10%)
                if alpha1 <= 0.0:
                    threshold = 2.0 # Keep nothing
                elif alpha1 >= 1.0:
                    threshold = -1.0 # Keep everything
                else:
                    threshold = torch.quantile(flat_probs, 1 - alpha1)
                
                pred_mask = flat_probs >= threshold
                pred_edges_hop1 = flat_edges[:, pred_mask]
                pred_types_hop1 = flat_types[pred_mask]
            else:
                pred_edges_hop1 = torch.empty((2, 0), device=self.device, dtype=torch.long)
                pred_types_hop1 = torch.empty((0,), device=self.device, dtype=torch.long)
        else:
            # 5. Concatenate the filtered results (The total size of this is much smaller than N^2 * R)
            if len(all_pred_edges_hop1) > 0:
                pred_edges_hop1 = torch.cat(all_pred_edges_hop1, dim=1)
                pred_types_hop1 = torch.cat(all_pred_types_hop1, dim=0)
            else:
                pred_edges_hop1 = torch.empty((2, 0), device=self.device, dtype=torch.long)
                pred_types_hop1 = torch.empty((0,), device=self.device, dtype=torch.long)

        # Calculate size1 and n_total based on the new structure
        size1 = pred_edges_hop1.size(1)
        n_total += num_candidates_hop1 * self.num_relations # Total predicted candidates check
        n_list.append(num_candidates_hop1 * self.num_relations)

        # 5. Compute Hop 1 Metrics (Set Comparison - Unchanged)
        true_triplets = set()
        for s, r, t in zip(subgraph_after.edge_index[0].tolist(), subgraph_after.edge_type.tolist(), subgraph_after.edge_index[1].tolist()):
            true_triplets.add((int(s), int(r), int(t)))

        pred_triplets = set()
        for s, r, t in zip(pred_edges_hop1[0].tolist(), pred_types_hop1.tolist(), pred_edges_hop1[1].tolist()):
            pred_triplets.add((int(s), int(r), int(t)))

        tp += len(pred_triplets & true_triplets)
        fp += len(pred_triplets - true_triplets)
        fn += len(true_triplets - pred_triplets)

        precision_hop1 = tp / (tp + fp + 1e-8)
        recall_hop1 = tp / (tp + fn + 1e-8)
        f1_hop1 = 2 * precision_hop1 * recall_hop1 / (precision_hop1 + recall_hop1 + 1e-8)
        size1 = len(pred_triplets)
        # n_total += probs.size(0)
        fail_number += fn

        # ==========================================
        # HOP 2: Conditional Prediction
        # ==========================================
        
        # Convert predicted triplets to string signatures for matching against trigger events (Unchanged)
        predicted_trigger_strings = set()
        for s, r, t in pred_triplets:
            try:
                s_name = id2ent[s]
                r_name = id2rel[r]
                t_name = id2ent[t]
                predicted_trigger_strings.add(f"{s_name} {r_name} {t_name}")
            except KeyError:
                continue

        f1_hop2_list, precision_hop2_list, recall_hop2_list = [], [], []
        size2 = 0

        # This loop is necessary to evaluate each conditional event.
        for added in newly_added_edges:
            # Construct the trigger string from the ground truth object
            trigger_list = added["trigger_event"][:3] # [Sub, Rel, Obj]
            trigger_str = " ".join(trigger_list)

            if trigger_str in predicted_trigger_strings:
                # --- TRIGGER PREDICTED: PROCEED TO HOP 2 ---
                
                matched_subgraph_after = added["subgraph_after"].to(self.device)
                matched_subgraph_before = added["subgraph_before"].to(self.device)
                matched_paragraph = added["paragraph"]
                
                # Update the input graph with the PREDICTED edges from Hop 1
                # Note: Assuming self.add_edges handles the concatenation correctly
                matched_subgraph_before = add_edges(matched_subgraph_before, pred_edges_hop1, pred_types_hop1)
                
                # Encode Hop 2 (Unchanged)
                if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                    z_text_2 = self.sentence_encoder.encode(matched_paragraph, convert_to_tensor=True).to(self.device)
                    if z_text_2.dim() == 2 and z_text_2.size(0) == 1: z_text_2 = z_text_2.squeeze(0)
                elif self.model_variant == 'rgat':
                    z_text_2 = self.sentence_encoder.encode(trigger_str, convert_to_tensor=True).to(self.device)
                    if z_text_2.dim() == 2 and z_text_2.size(0) == 1: z_text_2 = z_text_2.squeeze(0)
                else:
                    z_text_2 = None

                if self.model_variant in ['distmult', 'rgcn_baseline', 'transe', 'hitter', 'simkgc', 'kgt5']:
                    z2 = self.model.encode(matched_subgraph_before.x, matched_subgraph_before.edge_index, matched_subgraph_before.edge_type)
                else:
                    z2 = self.model.encode(matched_subgraph_before.x, matched_subgraph_before.edge_index, matched_subgraph_before.edge_type, z_text_2)

                # Generate Candidates Hop 2 (Unchanged)
                nodes_after_2 = torch.unique(torch.cat([matched_subgraph_after.edge_index[0], matched_subgraph_after.edge_index[1]]))
                head_mesh2, tail_mesh2 = torch.meshgrid(nodes_after_2, nodes_after_2, indexing='ij')
                edge_index_candidates2 = torch.stack([head_mesh2.flatten(), tail_mesh2.flatten()], dim=0)
                mask2 = edge_index_candidates2[0] != edge_index_candidates2[1]
                edge_index_candidates2 = edge_index_candidates2[:, mask2]

                if edge_index_candidates2.size(1) == 0:
                    f1_hop2_list.append(1.0) 
                    continue

                num_candidates_hop2 = edge_index_candidates2.size(1)

                # Decode Hop 2 (VECTORIZED OPTIMIZATION APPLIED HERE)
                
                # Repeat edges R times
                all_pred_edges2 = edge_index_candidates2.repeat(1, self.num_relations)

                # Create relation tensor: [0...0, 1...1, ..., R-1...R-1]
                rel_ids = torch.arange(self.num_relations, device=self.device)
                all_pred_types2 = rel_ids.repeat_interleave(num_candidates_hop2)
                    
                # Single massive decode call - SIGNIFICANT SPEEDUP
                all_pred_scores2 = self.model.decode(z2, all_pred_edges2, all_pred_types2)

                probs2 = torch.sigmoid(all_pred_scores2)
                probs2 = (probs2 - probs2.min()) / (probs2.max() - probs2.min() + 1e-8)
                
                # Apply Threshold Alpha 2
                if use_percentage:
                    if alpha2 <= 0.0:
                        threshold2 = 2.0
                    elif alpha2 >= 1.0:
                        threshold2 = -1.0
                    else:
                        threshold2 = torch.quantile(probs2, 1 - alpha2)
                    pred_mask2 = probs2 >= threshold2
                else:
                    pred_mask2 = probs2 >= alpha2

                pred_edges_hop2 = all_pred_edges2[:, pred_mask2]
                pred_types_hop2 = all_pred_types2[pred_mask2]

                # Hop 2 Metrics (Unchanged)
                true_triplets2 = set(zip(matched_subgraph_after.edge_index[0].tolist(), matched_subgraph_after.edge_type.tolist(), matched_subgraph_after.edge_index[1].tolist()))
                pred_triplets2 = set(zip(pred_edges_hop2[0].tolist(), pred_types_hop2.tolist(), pred_edges_hop2[1].tolist()))

                tp += len(pred_triplets2 & true_triplets2)
                fp += len(pred_triplets2 - true_triplets2)
                fn += len(true_triplets2 - pred_triplets2)
                # fail_number += fn2

                # prec2 = tp2 / (tp2 + fp2 + 1e-8)
                # rec2 = tp2 / (tp2 + fn2 + 1e-8)
                # f1_2 = 2 * prec2 * rec2 / (prec2 + rec2 + 1e-8)

                # f1_hop2_list.append(f1_2)
                # precision_hop2_list.append(prec2)
                # recall_hop2_list.append(rec2)
                size2 += len(pred_triplets2)
                n_total += probs2.size(0)
                n_list.append(probs2.size(0))
            else:
                # --- TRIGGER MISSED: HOP 2 FAILURE (Unchanged) ---
                f1_hop2_list.append(0.0)
                precision_hop2_list.append(0.0)
                recall_hop2_list.append(0.0)

        # ----------------
        # Aggregate metrics (Unchanged)
        # ----------------
        f1_hop2 = float(np.mean(f1_hop2_list)) if f1_hop2_list else 0.0
        precision_hop2 = float(np.mean(precision_hop2_list)) if precision_hop2_list else 0.0
        recall_hop2 = float(np.mean(recall_hop2_list)) if recall_hop2_list else 1.0 

        f1, recall, precision = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8), tp / (tp + fp + 1e-8)
        
        metrics = {
            "size1": size1,
            "size2": size2,
            "f1_hop2": f1_hop2,
            "precision_hop2": precision_hop2,
            "recall_hop2": recall_hop2,
            "f1_hop1": f1_hop1,
            "precision_hop1": precision_hop1,
            "recall_hop1": recall_hop1,
            "n_total": n_total,
            "n": tp + fn,
            "fail_number": fail_number,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

        return metrics
    
    @torch.no_grad()
    def _calibrate_aggregate(
        self, val_dataset 
    ):
        """
        Aggregate conformal calibration:
        - Aggregates nodes from all subgraph_after graphs in `newly_added_edges`
        - Predicts over all possible (s, r, t) among those nodes
        - Computes conformal quantile q_hat based on self.error_rate
        - Returns metrics and calibrated threshold q_hat
        """

        self.model.eval()

        q_hats = []
        recalls = []
        nonconformities = []

        for batch in val_dataset:
            b = batch[0]
            subgraph_before = b["subgraph_before"].to(self.device)
            newly_added_edges = b["newly_added_edges"]
            subgraph_after = b["subgraph_after"].to(self.device)
            subgraph_intermediate = b["subgraph_intermediate"].to(self.device)
            paragraph = b["paragraph"]
            trigger_event = b["trigger_event"]

            # ----------------------------------------------------
            # 1. Encode text if applicable
            # ----------------------------------------------------
            if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
                if z_text.dim() == 2 and z_text.size(0) == 1:
                    z_text = z_text.squeeze(0)
            elif self.model_variant == "rgat":
                # z_text = self.sentence_encoder.encode(
                #     " ".join([e[0] for e in trigger_event]), convert_to_tensor=True
                # ).to(self.device)
                z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
                if z_text.dim() == 2 and z_text.size(0) == 1:
                    z_text = z_text.squeeze(0)
            else:
                z_text = None

            # ----------------------------------------------------
            # 2. Encode base subgraph
            # ----------------------------------------------------
            if self.model_variant in ["distmult", "transe", "rgcn_baseline", "hitter", "simkgc", "kgt5"]:
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type)
            elif self.model_variant == "rgat":
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type, z_text)
            else:
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type, z_text)

            # ----------------------------------------------------
            # 3. Aggregate all nodes + true triplets from new edges
            # ----------------------------------------------------
            all_nodes = set()
            all_true_triplets = set()

            # Include Hop 1 targets (subgraph_after)
            all_nodes.update(subgraph_after.edge_index[0].tolist())
            all_nodes.update(subgraph_after.edge_index[1].tolist())
            for s, r, t in zip(
                subgraph_after.edge_index[0].tolist(),
                subgraph_after.edge_type.tolist(),
                subgraph_after.edge_index[1].tolist(),
            ):
                all_true_triplets.add((int(s), int(r), int(t)))

            for added in newly_added_edges:
                sub_after = added["subgraph_after"]
                src_nodes = sub_after.edge_index[0].tolist()
                dst_nodes = sub_after.edge_index[1].tolist()
                all_nodes.update(src_nodes)
                all_nodes.update(dst_nodes)

                # print(added)
                trigger_event = added["trigger_event"]
                all_true_triplets.add((self.ent2id[trigger_event[0]], self.rel2id[trigger_event[1]], self.ent2id[trigger_event[2]]))

                for s, r, t in zip(
                    sub_after.edge_index[0].tolist(),
                    sub_after.edge_type.tolist(),
                    sub_after.edge_index[1].tolist(),
                ):
                    all_true_triplets.add((int(s), int(r), int(t)))

            nodes_tensor = torch.tensor(sorted(list(all_nodes)), dtype=torch.long, device=self.device)

            # ----------------------------------------------------
            # 4. Generate candidate edges (Cartesian product)
            # ----------------------------------------------------
            head_mesh, tail_mesh = torch.meshgrid(nodes_tensor, nodes_tensor, indexing="ij")
            edge_index_candidates = torch.stack([head_mesh.flatten(), tail_mesh.flatten()], dim=0)
            mask = edge_index_candidates[0] != edge_index_candidates[1]
            edge_index_candidates = edge_index_candidates[:, mask]

            if edge_index_candidates.size(1) == 0:
                continue

            # ----------------------------------------------------
            # 5. Decode for all relations
            # ----------------------------------------------------
            all_pred_edges, all_pred_types, all_pred_scores = [], [], []

            for rel_id in range(self.num_relations):
                rel_tensor = torch.full(
                    (edge_index_candidates.size(1),),
                    rel_id,
                    dtype=torch.long,
                    device=self.device,
                )
                scores = self.model.decode(z1, edge_index_candidates, rel_tensor)
                all_pred_edges.append(edge_index_candidates)
                all_pred_types.append(rel_tensor)
                all_pred_scores.append(scores)

            all_pred_edges = torch.cat(all_pred_edges, dim=1)
            all_pred_types = torch.cat(all_pred_types, dim=0)
            all_pred_scores = torch.cat(all_pred_scores, dim=0)

            probs = torch.sigmoid(all_pred_scores)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

            # ----------------------------------------------------
            # 6. Compute conformal quantile q_hat
            # ----------------------------------------------------
            # Nonconformity = 1 - probability of true edge
            # We compute quantile over calibration scores of true triplets
            true_mask = torch.zeros_like(probs, dtype=torch.bool)
            triplet_idx_map = {
                (int(s), int(r), int(t)): idx
                for idx, (s, r, t) in enumerate(
                    zip(all_pred_edges[0].tolist(), all_pred_types.tolist(), all_pred_edges[1].tolist())
                )
            }

            for s, r, t in all_true_triplets:
                if (s, r, t) in triplet_idx_map:
                    true_mask[triplet_idx_map[(s, r, t)]] = True

            true_scores = probs[true_mask]
            if len(true_scores) == 0:
                q_hat = 1.0  # fallback if no calibration data
            else:
                nonconformity = 1.0 - true_scores
                q_hat = np.quantile(nonconformity.cpu().numpy(), 1 - self.error_rate) 
            
            nonconformities.extend(nonconformity.cpu().numpy())
            q_hats.append(q_hat)

            # ----------------------------------------------------
            # 7. Predict edges using calibrated threshold
            # ----------------------------------------------------
            pred_mask = (1.0 - probs) <= q_hat 
            pred_edges = all_pred_edges[:, pred_mask]
            pred_types = all_pred_types[pred_mask]

            pred_triplets = set(
                (int(s), int(r), int(t))
                for s, r, t in zip(
                    pred_edges[0].tolist(),
                    pred_types.tolist(),
                    pred_edges[1].tolist(),
                )
            )

            # ----------------------------------------------------
            # 8. Compute F1 / Precision / Recall
            # ----------------------------------------------------
            tp = len(pred_triplets & all_true_triplets)
            fp = len(pred_triplets - all_true_triplets)
            fn = len(all_true_triplets - pred_triplets)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            size = len(pred_triplets)

            coverage = tp / (len(all_true_triplets) + 1e-8)

            recalls.append(recall)

        print('recalls:', np.mean(recalls))
        q_hat = np.quantile(nonconformities, 1 - self.error_rate)
        # ----------------------------------------------------
        # 9. Return metrics and q_hat
        # ----------------------------------------------------
        metrics = {
            "F1": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
            "Size": int(size),
            "q_hat": float(q_hat),
            # "threshold": float(threshold),
        }

        return q_hat, np.mean(recalls)
    
    @torch.no_grad()
    def _evaluate_aggregate(self, test_dataset, q_hat):
        """
        Evaluate conformal prediction performance on test_dataset using calibrated q_hat.
        Uses the standard cascade prediction pipeline with a single threshold.
        """
        # Convert quantile to alpha (prob threshold)
        alpha = 1.0 - q_hat
        
        # Reuse the main evaluation function
        n, mean_f1, fail_rate_hat, avg_set_size, mean_precision, avg_fail_number = self._evaluate_alpha_combo_on_dataset(
            test_dataset, 
            self.device, 
            alpha1=alpha, 
            alpha2=alpha,
            use_percentage=False # Aggregate uses absolute thresholds
        )

        metrics = {
            "F1_macro": mean_f1,
            "Precision_macro": mean_precision,
            "Recall_macro": 1.0 - fail_rate_hat,
            "Coverage": 1.0 - fail_rate_hat, # In this context, coverage ~ recall
            "Avg_Set_Size": avg_set_size,
            "Threshold": alpha,
        }

        return mean_f1, mean_precision, 1.0 - fail_rate_hat, avg_set_size, 1.0 - fail_rate_hat

    def _fixed_sequence_accept(self, pvals_in_order, delta):
        """
        Accept sequentially at level delta, stop at first failure.
        Returns prefix length (#accepted).
        """
        acc = 0
        for p in pvals_in_order:
            print(p)
            if p <= delta:
                acc += 1 
        return acc 
    
    def _dominates(self, a, b):
        """
        Return True iff objective vector a dominates b (all <= and at least one <).
        We minimize all objectives.
        """
        le_all = all(x <= y for x, y in zip(a, b))
        lt_any = any(x < y for x, y in zip(a, b))
        return le_all and lt_any

    def _pareto_frontier(self,objs_with_cfgs):
        """
        objs_with_cfgs: list of (cfg, obj_tuple)
        where obj_tuple = (fail_rate_hat, avg_set_size[, ...]) minimized.
        Returns: list of cfgs on the Pareto frontier.
        """
        frontier = []
        for (cfg_i, obj_i) in objs_with_cfgs:
            dominated = False
            for (cfg_j, obj_j) in objs_with_cfgs:
                if cfg_i is cfg_j: 
                    continue
                if self._dominates(obj_j, obj_i):
                    dominated = True
                    break
            if not dominated:
                frontier.append((cfg_i, obj_i))
        return frontier

    def _evaluate_alpha_combo_on_dataset(
        self, dataset, device, alpha1, alpha2, use_percentage=False 
    ):
        f1_scores, precisions, set_sizes, recalls, fail_numbers = [], [], [], [], []
        n = 0 

        for batch in dataset:
            b = batch[0] if isinstance(batch, (list, tuple)) else batch

            try:
                metrics = self._forward_cascade_prediction(
                    newly_added_edges=b["newly_added_edges"],
                    subgraph_before=b["subgraph_before"].to(device),
                    subgraph_after=b["subgraph_after"].to(device),
                    subgraph_intermediate=b["subgraph_intermediate"].to(device),
                    trigger_event=b["trigger_event"],
                    paragraph=b["paragraph"],
                    alpha1=alpha1,
                    alpha2=alpha2,
                    id2ent= self.id2ent,
                    id2rel= self.id2rel,
                    use_percentage=use_percentage
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # Catch OOM errors when alpha is too low (causing too many edges)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    # Return worst-case metrics to ensure this configuration is rejected
                    # (n, f1, fail_rate, set_size, precision, fail_number)
                    return 0, 0.0, 1.0, float('inf'), 0.0, float('inf')
                raise e


            set_size = metrics["size1"] + metrics["size2"]
            n += metrics["n"]

            f1_scores.append(metrics["f1"])
            precisions.append(metrics["precision"])
            set_sizes.append(set_size)
            recalls.append(metrics["recall"])
            fail_numbers.append(metrics["fail_number"])

        mean_f1 = float(np.mean(f1_scores))
        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))
        avg_set_size = float(np.mean(set_sizes))
        avg_fail_number = float(np.mean(fail_numbers))
        fail_rate_hat = 1 - mean_recall 

        return n, mean_f1, fail_rate_hat, avg_set_size, mean_precision, avg_fail_number

    def calibrate_ltt_pareto(self):
        # ----- Step 1: Enumerate α₁, α₂ combinations -----
        # might need model specific adjustment
        alphas = np.arange(0, 1.0 + 1e-9, self.grid_step)
        combinations = list(product(alphas, alphas))

        # ---- Split Dataset into Optimization and Calibration -----
        # n = len(self.cal_dataset)
        # n_opt = int(n * self.split_ratio)
        # n_t = n - n_opt # number of final test samples

        batches = list(self.cal_dataset)
        # 1. Get total indices
        indices = list(range(len(batches)))

        # 2. Shuffle indices (rnd is your random.Random object)
        random.seed(19)
        random.shuffle(indices)

        # 3. Use indices to select your D_opt and D_test sets
        cut_idx = max(1, int(len(indices) * self.split_ratio))
        opt_indices = indices[:cut_idx]
        test_indices = indices[cut_idx:]

        # 4. Create the final datasets (list comprehension is cleaner/safer than splicing)
        D_opt = [batches[i] for i in opt_indices]
        D_test = [batches[i] for i in test_indices]


        def calculate_average_complexity(dataset):
            """Calculates the average number of nodes and edges across all graphs in the dataset."""
            node_counts = []
            edge_counts = []
            
            for batch in dataset:
                # Assuming the structure is: [batch_data, ...] where batch_data is a dict
                b = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                # Access the complexity measure (using subgraph_before as the baseline)
                graph = b["subgraph_after"]
                
                # Collect counts
                node_counts.append(graph.num_nodes)
                edge_counts.append(graph.num_edges)

            if not node_counts:
                return 0, 0
            
            avg_nodes = np.mean(node_counts)
            avg_edges = np.mean(edge_counts)
            
            return avg_nodes, avg_edges

        # Assuming D_opt and D_test have already been defined:
        # D_opt = [batches[i] for i in opt_indices]
        # D_test = [batches[i] for i in test_indices]

        # Run calculations
        avg_nodes_opt, avg_edges_opt = calculate_average_complexity(D_opt)
        avg_nodes_test, avg_edges_test = calculate_average_complexity(D_test)


        print("\n--- Average Graph Complexity ---")
        print(f"D_opt Average Nodes: {avg_nodes_opt:.2f}")
        print(f"D_opt Average Edges: {avg_edges_opt:.2f}")
        print("--------------------------------")
        print(f"D_test Average Nodes: {avg_nodes_test:.2f}")
        print(f"D_test Average Edges: {avg_edges_test:.2f}")

        # stage_1: evalaute on D_opt with pareto frontier
        stage1_objs = []   # [(cfg, (fail_rate_hat, avg_set_size))]
        stage1_pvals = {}  # configuration -> p_opt

        for (a1, a2) in combinations:
            n_opt, mean_f1_opt, fail_rate_hat_opt, avg_size_opt, _, avg_fail_number_opt = self._evaluate_alpha_combo_on_dataset(
                D_opt, self.device, a1, a2, use_percentage=True
            )

            k_opt = int(math.floor(n_opt * fail_rate_hat_opt))

            p_opt = float(binom.cdf(k_opt, int(n_opt), self.error_rate))

            stage1_objs.append(((a1, a2), [fail_rate_hat_opt]))
            stage1_pvals[(a1, a2)] = p_opt

            print(f"[OPT] α1={a1:.2f}, α2={a2:.2f} | F1={mean_f1_opt:.3f}"
              f"fail={fail_rate_hat_opt:.3f} size={avg_size_opt:.2f} p_opt={p_opt:.4f} (k={k_opt}/{n_opt})")

        # frontier = self._pareto_frontier(stage1_objs)
        ordered_frontier = sorted(stage1_objs, key=lambda x: x[1][0])
        # ordered_frontier = sorted(frontier, key=lambda x: x[1][0])

        # stage 2: fixed sequence on the calibration split
        stage2_rows = []
        p_tests = []
        for item in ordered_frontier:
            a1, a2 = item[0]
            n_t, mean_f1_t, fail_rate_hat_t, avg_size_t, _, avg_fail_number_t = self._evaluate_alpha_combo_on_dataset(
                D_test, self.device, a1, a2, use_percentage=True
            )
            k_t = int(math.floor(n_t * fail_rate_hat_t))
            p_t = float(binom.cdf(k_t, int(n_t), self.error_rate))
            stage2_rows.append({
                "alpha1": a1, "alpha2": a2,
                "n_test": n_t, "mean_f1": mean_f1_t,
                "fail_rate_hat": fail_rate_hat_t,
                "avg_set_size": avg_size_t,
                "p_test": p_t
            })
            p_tests.append(p_t)
            print(f"[TEST] α1={a1:.2f}, α2={a2:.2f} | F1={mean_f1_t:.3f} "
               f"fail={fail_rate_hat_t:.3f} size={avg_size_t:.2f} p_test={p_t:.4f} (k={k_t}/{n_t})")

        # Fixed-sequence accept
        # num_accepted = self._fixed_sequence_accept(p_tests, self.delta)
        # accepted_rows = stage2_rows[:num_accepted]

        # Filter by p-value directly (ignoring sequence order as requested)
        accepted_rows = [row for row in stage2_rows if row['p_test'] <= self.delta]
        
        if len(accepted_rows) == 0:
            best_emp = min(stage2_rows, key=lambda r: (r["avg_set_size"]))
            print("⚠️  No configuration passed the fixed-sequence test at δ. "
                "Returning best empirical (tie-break by smaller set size).")
            return {
                "status": "no_valid",
                "ordered_frontier": ordered_frontier,
                "accepted": [],
                "best_empirical": best_emp,
                "stage2_table": stage2_rows
            }

        # Among the accepted prefix, pick the one minimizing avg_set_size
        best = min(accepted_rows, key=lambda r: r["avg_set_size"])
        print(f"✅ Selected α1={best['alpha1']:.2f}, α2={best['alpha2']:.2f} "
            f"(avg_size={best['avg_set_size']:.2f}, F1={best['mean_f1']:.3f}, p_test={best['p_test']:.4f})")

        return {
            "status": "ok",
            "best_config": best,
            "ordered_frontier": ordered_frontier,
            "accepted": accepted_rows,
            # "stage1_popts": {cfg: stage1_pvals[cfg] for cfg in ordered_frontier},
            "stage2_table": stage2_rows
        }
    
    @torch.no_grad()
    def baseline_calibration(self, val_dataset):
        self.model.eval()
        scores_hop1, scores_hop2 = [], []

        for batch in val_dataset:
            b = batch[0]
            subgraph_before = b["subgraph_before"].to(self.device)
            newly_added_edges = b["newly_added_edges"]
            subgraph_after = b["subgraph_after"].to(self.device)
            paragraph = b["paragraph"]
            trigger_event = b["trigger_event"]

            # =========================================
            # HOP 1: Predict on subgraph_after
            # =========================================
            
            # 1. Encode Z1
            if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
                if z_text.dim() == 2 and z_text.size(0) == 1: z_text = z_text.squeeze(0)
            elif self.model_variant == 'rgat':
                z_text = self.sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(self.device)
                if z_text.dim() == 2 and z_text.size(0) == 1: z_text = z_text.squeeze(0)
            else:
                z_text = None

            if self.model_variant in ['distmult', 'rgcn_baseline', 'transe', 'hitter', 'simkgc', 'kgt5']:
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type)
            else:
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type, z_text)

            # ---------------------------------------------------------
            # CHANGE 1: Use Global Candidate Generation (Match Evaluation)
            # ---------------------------------------------------------
            # We must include nodes from Hop 2 in the candidate pool for Hop 1
            # so the model sees the same "distractors" during calibration.
            all_nodes_1 = set(subgraph_after.edge_index.unique().tolist())
            for added in newly_added_edges:
                all_nodes_1.update(added["subgraph_after"].edge_index.unique().tolist())

            nodes_tensor_1 = torch.tensor(sorted(list(all_nodes_1)), dtype=torch.long, device=self.device)

            # 3. Generate Candidates (Cartesian Product)
            head_mesh, tail_mesh = torch.meshgrid(nodes_tensor_1, nodes_tensor_1, indexing="ij")
            edge_index_candidates = torch.stack([head_mesh.flatten(), tail_mesh.flatten()], dim=0)
            mask = edge_index_candidates[0] != edge_index_candidates[1]
            edge_index_candidates = edge_index_candidates[:, mask]

            if edge_index_candidates.size(1) > 0:
                # 4. Decode all relations
                all_pred_edges, all_pred_types, all_pred_scores = [], [], []
                for rel_id in range(self.num_relations):
                    rel_tensor = torch.full((edge_index_candidates.size(1),), rel_id, dtype=torch.long, device=self.device)
                    scores = self.model.decode(z1, edge_index_candidates, rel_tensor)
                    
                    all_pred_edges.append(edge_index_candidates)
                    all_pred_types.append(rel_tensor)
                    all_pred_scores.append(scores)

                all_pred_edges = torch.cat(all_pred_edges, dim=1)
                all_pred_types = torch.cat(all_pred_types, dim=0)
                all_pred_scores = torch.cat(all_pred_scores, dim=0)

                # 5. Normalize
                probs = torch.sigmoid(all_pred_scores)
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

                # 6. Extract Scores for True Triplets (Nonconformity)
                true_triplets_1 = set()
                for s, r, t in zip(subgraph_after.edge_index[0].tolist(), 
                                   subgraph_after.edge_type.tolist(), 
                                   subgraph_after.edge_index[1].tolist()):
                    true_triplets_1.add((int(s), int(r), int(t)))

                true_mask = torch.zeros_like(probs, dtype=torch.bool)
                triplet_idx_map = {
                    (int(s), int(r), int(t)): idx 
                    for idx, (s, r, t) in enumerate(zip(all_pred_edges[0].tolist(), all_pred_types.tolist(), all_pred_edges[1].tolist()))
                }

                for s, r, t in true_triplets_1:
                    if (s, r, t) in triplet_idx_map:
                        true_mask[triplet_idx_map[(s, r, t)]] = True
                
                true_scores = probs[true_mask]
                if len(true_scores) > 0:
                    scores_hop1.extend((1.0 - true_scores).cpu().tolist())

            # =========================================
            # HOP 2: Predict on newly_added_edges
            # =========================================
            # Note: We keep this independent (clean graph input) because we don't 
            # have a threshold to generate predicted noise yet.
            for added in newly_added_edges:
                subgraph_before_2 = added["subgraph_before"].to(self.device)
                subgraph_after_2 = added["subgraph_after"].to(self.device)
                paragraph_2 = added["paragraph"]
                trigger_event_2 = added["trigger_event"]

                # 1. Encode Z2
                if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                    z_text_2 = self.sentence_encoder.encode(paragraph_2, convert_to_tensor=True).to(self.device)
                    if z_text_2.dim() == 2 and z_text_2.size(0) == 1: z_text_2 = z_text_2.squeeze(0)
                elif self.model_variant == 'rgat':
                    z_text_2 = self.sentence_encoder.encode(" ".join([e[0] for e in trigger_event_2]), convert_to_tensor=True).to(self.device)
                    if z_text_2.dim() == 2 and z_text_2.size(0) == 1: z_text_2 = z_text_2.squeeze(0)
                else:
                    z_text_2 = None

                if self.model_variant in ['distmult', 'rgcn_baseline', 'transe', 'hitter', 'simkgc', 'kgt5']:
                    z2 = self.model.encode(subgraph_before_2.x, subgraph_before_2.edge_index, subgraph_before_2.edge_type)
                else:
                    z2 = self.model.encode(subgraph_before_2.x, subgraph_before_2.edge_index, subgraph_before_2.edge_type, z_text_2)

                # 2. Identify Nodes (Local is fine for Hop 2, same as Eval)
                all_nodes_2 = set(subgraph_after_2.edge_index[0].tolist() + subgraph_after_2.edge_index[1].tolist())
                nodes_tensor_2 = torch.tensor(sorted(list(all_nodes_2)), dtype=torch.long, device=self.device)

                # 3. Generate Candidates
                head_mesh_2, tail_mesh_2 = torch.meshgrid(nodes_tensor_2, nodes_tensor_2, indexing="ij")
                edge_index_candidates_2 = torch.stack([head_mesh_2.flatten(), tail_mesh_2.flatten()], dim=0)
                mask_2 = edge_index_candidates_2[0] != edge_index_candidates_2[1]
                edge_index_candidates_2 = edge_index_candidates_2[:, mask_2]

                if edge_index_candidates_2.size(1) == 0:
                    continue

                # 4. Decode
                all_pred_edges, all_pred_types, all_pred_scores = [], [], []
                for rel_id in range(self.num_relations):
                    rel_tensor = torch.full((edge_index_candidates_2.size(1),), rel_id, dtype=torch.long, device=self.device)
                    scores = self.model.decode(z2, edge_index_candidates_2, rel_tensor)
                    all_pred_edges.append(edge_index_candidates_2)
                    all_pred_types.append(rel_tensor)
                    all_pred_scores.append(scores)

                all_pred_edges = torch.cat(all_pred_edges, dim=1)
                all_pred_types = torch.cat(all_pred_types, dim=0)
                all_pred_scores = torch.cat(all_pred_scores, dim=0)

                # 5. Normalize
                probs = torch.sigmoid(all_pred_scores)
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

                # 6. Extract Scores for True Triplets
                true_triplets_2 = set()
                for s, r, t in zip(subgraph_after_2.edge_index[0].tolist(), 
                                   subgraph_after_2.edge_type.tolist(), 
                                   subgraph_after_2.edge_index[1].tolist()):
                    true_triplets_2.add((int(s), int(r), int(t)))
                try:
                    true_triplets_2.add((self.ent2id[trigger_event_2[0]], 
                                         self.rel2id[trigger_event_2[1]], 
                                         self.ent2id[trigger_event_2[2]]))
                except (KeyError, AttributeError):
                    pass

                true_mask = torch.zeros_like(probs, dtype=torch.bool)
                triplet_idx_map = {
                    (int(s), int(r), int(t)): idx 
                    for idx, (s, r, t) in enumerate(zip(all_pred_edges[0].tolist(), all_pred_types.tolist(), all_pred_edges[1].tolist()))
                }

                for s, r, t in true_triplets_2:
                    if (s, r, t) in triplet_idx_map:
                        true_mask[triplet_idx_map[(s, r, t)]] = True
                
                true_scores = probs[true_mask]
                if len(true_scores) > 0:
                    scores_hop2.extend((1.0 - true_scores).cpu().tolist())

        # Compute Bonferroni-corrected quantiles
        q1_hat = np.quantile(scores_hop1, 1 - self.error_rate / 2) if scores_hop1 else 1.0
        q2_hat = np.quantile(scores_hop2, 1 - self.error_rate / 2) if scores_hop2 else 1.0

        return q1_hat, q2_hat
    
    @torch.no_grad()
    def baseline_evaluation(self, test_dataset, q1_hat, q2_hat):
        """
        Modified Baseline Evaluation to match Cascade Logic:
        1. Uses global candidate space for Hop 1 (Union of nodes).
        2. Enforces Conditional Constraint: Hop 2 only runs if Trigger is predicted.
        3. Propagates predicted edges from Hop 1 to Hop 2.
        """
        self.model.eval()
        
        # Convert nonconformity quantiles to probability thresholds
        # score >= alpha means we accept
        alpha1 = 1.0 - q1_hat
        alpha2 = 1.0 - q2_hat

        f1_scores, precisions, recalls, set_sizes = [], [], [], []
        fail_rate_list = []

        for batch in test_dataset:
            b = batch[0]
            subgraph_before = b["subgraph_before"].to(self.device)
            newly_added_edges = b["newly_added_edges"]
            subgraph_after = b["subgraph_after"].to(self.device)
            paragraph = b["paragraph"]
            trigger_event = b["trigger_event"]

            cur_size = 0
            
            # =========================================
            # HOP 1: Global Prediction (Same as Cascade)
            # =========================================
            
            # 1. Encode Z1
            if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
                if z_text.dim() == 2 and z_text.size(0) == 1: z_text = z_text.squeeze(0)
            elif self.model_variant == 'rgat':
                z_text = self.sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(self.device)
                if z_text.dim() == 2 and z_text.size(0) == 1: z_text = z_text.squeeze(0)
            else:
                z_text = None

            if self.model_variant in ['distmult', 'rgcn_baseline', 'transe', 'hitter', 'simkgc', 'kgt5']:
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type)
            else:
                z1 = self.model.encode(subgraph_before.x, subgraph_before.edge_index, subgraph_before.edge_type, z_text)

            # 2. Generate Candidates (Global Union - Logic matched to Cascade)
            nodes_after = set(subgraph_after.edge_index.unique().tolist())
            for added in newly_added_edges:
                nodes_after.update(added["subgraph_after"].edge_index.unique().tolist())
            
            nodes_tensor_1 = torch.tensor(sorted(list(nodes_after)), dtype=torch.long, device=self.device)

            head_mesh, tail_mesh = torch.meshgrid(nodes_tensor_1, nodes_tensor_1, indexing="ij")
            edge_index_candidates = torch.stack([head_mesh.flatten(), tail_mesh.flatten()], dim=0)
            mask = edge_index_candidates[0] != edge_index_candidates[1]
            edge_index_candidates = edge_index_candidates[:, mask]

            # 3. Decode & Threshold
            if edge_index_candidates.size(1) > 0:
                all_pred_edges, all_pred_types, all_pred_scores = [], [], []
                for rel_id in range(self.num_relations):
                    rel_tensor = torch.full((edge_index_candidates.size(1),), rel_id, dtype=torch.long, device=self.device)
                    scores = self.model.decode(z1, edge_index_candidates, rel_tensor)
                    all_pred_edges.append(edge_index_candidates)
                    all_pred_types.append(rel_tensor)
                    all_pred_scores.append(scores)

                all_pred_edges = torch.cat(all_pred_edges, dim=1)
                all_pred_types = torch.cat(all_pred_types, dim=0)
                all_pred_scores = torch.cat(all_pred_scores, dim=0)

                probs = torch.sigmoid(all_pred_scores)
                # Normalize exactly as in Cascade
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

                # APPLY THRESHOLD ALPHA 1
                pred_mask = probs >= alpha1
                
                pred_edges_hop1 = all_pred_edges[:, pred_mask]
                pred_types_hop1 = all_pred_types[pred_mask]
            else:
                pred_edges_hop1 = torch.empty((2,0), device=self.device)
                pred_types_hop1 = torch.empty((0,), device=self.device)

            # 4. Metrics Hop 1
            pred_triplets_hop1 = set()
            if pred_edges_hop1.size(1) > 0:
                for s, r, t in zip(pred_edges_hop1[0].tolist(), pred_types_hop1.tolist(), pred_edges_hop1[1].tolist()):
                    pred_triplets_hop1.add((int(s), int(r), int(t)))

            true_triplets_hop1 = set()
            for s, r, t in zip(subgraph_after.edge_index[0].tolist(), subgraph_after.edge_type.tolist(), subgraph_after.edge_index[1].tolist()):
                true_triplets_hop1.add((int(s), int(r), int(t)))

            tp = len(pred_triplets_hop1 & true_triplets_hop1)
            fp = len(pred_triplets_hop1 - true_triplets_hop1)
            fn = len(true_triplets_hop1 - pred_triplets_hop1)
            
            cur_size += len(pred_triplets_hop1)
            
            # Store Hop 1 Metrics temporarily if you want, but we usually aggregate everything
            prec_h1 = tp / (tp + fp + 1e-8)
            rec_h1 = tp / (tp + fn + 1e-8)
            f1_h1 = 2 * prec_h1 * rec_h1 / (prec_h1 + rec_h1 + 1e-8)
            
            # =========================================
            # HOP 2: Conditional Prediction
            # =========================================
            
            # Prepare predicted strings for trigger checking
            predicted_trigger_strings = set()
            for s, r, t in pred_triplets_hop1:
                try:
                    s_name = self.id2ent[s]
                    r_name = self.id2rel[r]
                    t_name = self.id2ent[t]
                    predicted_trigger_strings.add(f"{s_name} {r_name} {t_name}")
                except KeyError:
                    continue

            hop2_f1s, hop2_precs, hop2_recs = [], [], []

            for added in newly_added_edges:
                trigger_list = added["trigger_event"][:3]
                trigger_str = " ".join(trigger_list)

                # --- CONDITIONAL CHECK ---
                if trigger_str in predicted_trigger_strings:
                    # TRIGGER FOUND: PROCEED
                    
                    matched_subgraph_before = added["subgraph_before"].to(self.device)
                    matched_subgraph_after = added["subgraph_after"].to(self.device)
                    matched_paragraph = added["paragraph"]
                    
                    # !!! CRITICAL: Update Graph with Predicted Edges from Hop 1 !!!
                    matched_subgraph_before = add_edges(matched_subgraph_before, pred_edges_hop1, pred_types_hop1)

                    # Encode Hop 2
                    if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                        z_text_2 = self.sentence_encoder.encode(matched_paragraph, convert_to_tensor=True).to(self.device)
                        if z_text_2.dim() == 2 and z_text_2.size(0) == 1: z_text_2 = z_text_2.squeeze(0)
                    elif self.model_variant == 'rgat':
                        z_text_2 = self.sentence_encoder.encode(trigger_str, convert_to_tensor=True).to(self.device)
                        if z_text_2.dim() == 2 and z_text_2.size(0) == 1: z_text_2 = z_text_2.squeeze(0)
                    else:
                        z_text_2 = None

                    if self.model_variant in ['distmult', 'rgcn_baseline', 'transe', 'hitter', 'simkgc', 'kgt5']:
                        z2 = self.model.encode(matched_subgraph_before.x, matched_subgraph_before.edge_index, matched_subgraph_before.edge_type)
                    else:
                        z2 = self.model.encode(matched_subgraph_before.x, matched_subgraph_before.edge_index, matched_subgraph_before.edge_type, z_text_2)

                    # Generate Candidates (Local for Hop 2, as per Cascade)
                    nodes_after_2 = torch.unique(torch.cat([matched_subgraph_after.edge_index[0], matched_subgraph_after.edge_index[1]]))
                    head_mesh2, tail_mesh2 = torch.meshgrid(nodes_after_2, nodes_after_2, indexing='ij')
                    edge_index_candidates2 = torch.stack([head_mesh2.flatten(), tail_mesh2.flatten()], dim=0)
                    mask2 = edge_index_candidates2[0] != edge_index_candidates2[1]
                    edge_index_candidates2 = edge_index_candidates2[:, mask2]

                    if edge_index_candidates2.size(1) == 0:
                         hop2_f1s.append(1.0)
                         hop2_precs.append(1.0)
                         hop2_recs.append(1.0)
                         continue

                    # Decode
                    all_pred_edges2 = edge_index_candidates2.repeat(1, self.num_relations)
                    rel_ids = torch.arange(self.num_relations, device=self.device)
                    all_pred_types2 = rel_ids.repeat_interleave(edge_index_candidates2.size(1))
                    
                    scores2 = self.model.decode(z2, all_pred_edges2, all_pred_types2)
                    probs2 = torch.sigmoid(scores2)
                    probs2 = (probs2 - probs2.min()) / (probs2.max() - probs2.min() + 1e-8)

                    # APPLY THRESHOLD ALPHA 2
                    pred_mask2 = probs2 >= alpha2
                    
                    pred_edges_hop2 = all_pred_edges2[:, pred_mask2]
                    pred_types_hop2 = all_pred_types2[pred_mask2]

                    # Metrics Hop 2
                    true_triplets2 = set(zip(matched_subgraph_after.edge_index[0].tolist(), matched_subgraph_after.edge_type.tolist(), matched_subgraph_after.edge_index[1].tolist()))
                    pred_triplets2 = set(zip(pred_edges_hop2[0].tolist(), pred_types_hop2.tolist(), pred_edges_hop2[1].tolist()))

                    tp2 = len(pred_triplets2 & true_triplets2)
                    fp2 = len(pred_triplets2 - true_triplets2)
                    fn2 = len(true_triplets2 - pred_triplets2)

                    hop2_precs.append(tp2 / (tp2 + fp2 + 1e-8))
                    hop2_recs.append(tp2 / (tp2 + fn2 + 1e-8))
                    hop2_f1s.append(2 * (tp2 / (tp2 + fp2 + 1e-8)) * (tp2 / (tp2 + fn2 + 1e-8)) / ((tp2 / (tp2 + fp2 + 1e-8)) + (tp2 / (tp2 + fn2 + 1e-8)) + 1e-8))
                    
                    cur_size += len(pred_triplets2)

                else:
                    # --- TRIGGER MISSED: FAILURE ---
                    hop2_f1s.append(0.0)
                    hop2_precs.append(0.0)
                    hop2_recs.append(0.0)
            
            # Aggregate per batch (Macro over hops)
            mean_f1_h2 = np.mean(hop2_f1s) if hop2_f1s else 0.0
            mean_prec_h2 = np.mean(hop2_precs) if hop2_precs else 0.0
            mean_rec_h2 = np.mean(hop2_recs) if hop2_recs else 1.0 # If no hop 2 exists, recall is effectively 1 or N/A

            # Combine Hop 1 and Hop 2
            # Here we just take the macro average of the two stages per sample
            # (Adjust weighting if your original paper does otherwise)
            if len(newly_added_edges) > 0:
                 batch_f1 = (f1_h1 + mean_f1_h2) / 2
                 batch_prec = (prec_h1 + mean_prec_h2) / 2
                 batch_rec = (rec_h1 + mean_rec_h2) / 2
            else:
                 batch_f1 = f1_h1
                 batch_prec = prec_h1
                 batch_rec = rec_h1
            
            f1_scores.append(batch_f1)
            precisions.append(batch_prec)
            recalls.append(batch_rec)
            set_sizes.append(cur_size)

        # Final Aggregation
        mean_f1 = float(np.mean(f1_scores))
        mean_prec = float(np.mean(precisions))
        mean_rec = float(np.mean(recalls))
        mean_size = float(np.mean(set_sizes))
        fail_rate = 1.0 - mean_rec

        # Return format matching original baseline
        return mean_f1, mean_prec, mean_rec, 0.0, fail_rate, mean_size