from tqdm import tqdm, trange
import torch
from utils.utils import negative_sampling, compute_mrr
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import wandb
import gc
import os

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        batch_size, 
        device, 
        model_variant, 
        sentence_encoder, 
        save_path,
        num_epochs,
        test_data_loader):
        self.model = model 
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.model_variant = model_variant
        self.sentence_encoder = sentence_encoder
        self.save_path = save_path
        self.num_epochs = num_epochs

    def train_one_epoch(self, train_data_loader):
        self.model.train()

        for i, batch in enumerate(tqdm(train_data_loader, total=len(train_data_loader))):
            # Test
            # if i >= 2:
            #     continue

            self.optimizer.zero_grad()
            trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch

            paragraph = "; ".join(paragraph)
            # Transfer to GPU
            subgraph_orig = subgraph_orig.to(self.device)
            subgraph_mod_init = subgraph_mod_init.to(self.device)
            subgraph_mod = subgraph_mod.to(self.device)

            if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                with torch.no_grad():
                    z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
                    if z_text.dim() == 2 and z_text.size(0) == 1:
                        z_text = z_text.squeeze(0)

            if self.model_variant == 'rgat':
                with torch.no_grad():
                    z_text = self.sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(self.device)
                    if z_text.dim() == 2 and z_text.size(0) == 1:
                        z_text = z_text.squeeze(0)

            # Encode
            if self.model_variant == 'distmult':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'rgcn_baseline':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'rgat':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'transe':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'hitter':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'compgcn':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'simkgc':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'kgt5':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            else:
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text)

            z_cur = z

            pos_out = self.model.decode(z_cur, subgraph_mod.edge_index, subgraph_mod.edge_type)

            neg_edge_index = negative_sampling(subgraph_mod.edge_index, subgraph_mod.num_nodes)
            neg_out = self.model.decode(z_cur, neg_edge_index, subgraph_mod.edge_type)

            out = torch.cat([pos_out, neg_out])
            gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
            loss = F.binary_cross_entropy_with_logits(out, gt)

            wandb.log({"loss": loss.item()})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            del subgraph_orig, subgraph_mod_init, subgraph_mod
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def test(self, test_data_loader):
        self.model.eval()
        total_mrr = 0
        all_preds, all_labels = [], []

        for batch in tqdm(test_data_loader, total=len(test_data_loader)//self.batch_size): 
            trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch

            paragraph = "; ".join(paragraph)

            # Put things on GPU
            subgraph_orig = subgraph_orig.to(self.device)
            subgraph_mod_init = subgraph_mod_init.to(self.device)
            subgraph_mod = subgraph_mod.to(self.device)


            if self.model_variant in ('text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film'):
                with torch.no_grad():
                    z_text = self.sentence_encoder.encode(paragraph, convert_to_tensor=True).to(self.device)
                    if z_text.dim() == 2 and z_text.size(0) == 1:
                        z_text = z_text.squeeze(0)

            if self.model_variant == 'rgat':
                with torch.no_grad():
                    z_text = self.sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(self.device)
                    if z_text.dim() == 2 and z_text.size(0) == 1:
                        z_text = z_text.squeeze(0)

            # Encode
            if self.model_variant == 'distmult':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'rgcn_baseline':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'rgat':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'transe':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'hitter':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'compgcn':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'simkgc':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            elif self.model_variant == 'kgt5':
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            else:
                z = self.model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text)

            # Compute MRR
            mrr = compute_mrr(z, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, self.model)
            total_mrr += mrr

            z_cur = z

            pos_out = self.model.decode(z_cur, subgraph_mod.edge_index, subgraph_mod.edge_type)
            neg_edge_index = negative_sampling(subgraph_mod.edge_index, subgraph_mod.num_nodes)
            neg_out = self.model.decode(z_cur, neg_edge_index, subgraph_mod.edge_type)

            out = torch.cat([pos_out, neg_out])
            labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
            preds = (torch.sigmoid(out) > 0.5).long()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        f1 = f1_score(all_labels.numpy(), all_preds.numpy())
        precision = precision_score(all_labels.numpy(), all_preds.numpy())
        recall = recall_score(all_labels.numpy(), all_preds.numpy())
        roc_auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())

        return total_mrr / len(test_data_loader), f1, precision, recall, roc_auc, accuracy 
    
    def train(self, train_data_loader, val_data_loader, test_data_loader):
        self.model.train()

        if os.path.exists(self.save_path):
            print(f"Found existing model at {self.save_path}")
            self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
            self.model = self.model.to(self.device)
            print("Model loaded successfully!")
            return self.model
        else:
            print(f"No existing model found at {self.save_path}")
            print("Starting training from scratch...")
       
            best_val_mrr = 0
            best_epoch = 0

            for epoch in trange(self.num_epochs, desc="Epochs", ncols=80):
                print(f"Start Epoch: {epoch}")

                self.train_one_epoch(train_data_loader)
                val_mrr, val_f1, val_precision, val_recall, val_roc_auc, val_acc = self.test(val_data_loader)

                wandb.log({
                    "val_mrr": val_mrr,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_roc_auc": val_roc_auc,
                    "val_acc": val_acc
                })

                if val_mrr >= best_val_mrr:
                    best_val_mrr = val_mrr
                    test_mrr, test_f1, test_precision, test_recall, test_roc_auc, test_acc = self.test(test_data_loader)
                    final_test_mrr = test_mrr
                    final_test_f1 = test_f1
                    final_test_precision = test_precision
                    final_test_recall = test_recall
                    final_test_acc = test_acc
                    best_epoch = epoch

                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Update best model at epoch {epoch} with val_mrr {val_mrr:.4f}, "
                          f"test_mrr {test_mrr:.4f}, test_f1 {test_f1:.4f}, "
                          f"test_precision {test_precision:.4f}, test_recall {test_recall:.4f}, test_roc_auc {test_roc_auc:.4f}, test_acc {test_acc:.4f}")

                print(f"Epoch: {epoch:03d}, Val: {val_mrr:.4f}, Test: {test_mrr:.4f}")
                print(f"Best Epoch: {best_epoch}, Final Test MRR: {final_test_mrr:.4f}, "
                      f"Final Test F1: {final_test_f1:.4f}")

        return self.model