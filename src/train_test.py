import os, gc, time, copy, random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import json
from datetime import datetime
import re
from .kgbuilder import KnowledgeGraphBuilder   
from .encoder import herb_data_dict, build_text_nes_features
from .maged import MAGED, compute_target_loss, evaluate_model, save_loss_plot, FusionEncoder
from .utils_embedding import (
    load_node_embeddings, build_global_pos_dict, batch_entity_names, build_popularity_weights,
    get_train_loader, get_val_loader
)

REPO_ROOT = Path(__file__).resolve().parents[1]
def D(*parts: str) -> str:
    return str((REPO_ROOT / Path(*parts)).resolve())
EDGE_KEY_RE = re.compile(r"^gnn\.convs\.\d+\.convs\.<([^>]+)>\.")
def extract_edge_types_from_state(state_dict: dict):
    seen = set()
    out = []
    for k in state_dict.keys():
        m = EDGE_KEY_RE.match(k)
        if m:
            trip = m.group(1)  # e.g. "TCM_ID___herb_modulates_target___Protein"
            parts = trip.split("___")
            if len(parts) == 3:
                tup = (str(parts[0]), str(parts[1]), str(parts[2]))
                if tup not in seen:
                    seen.add(tup)
                    out.append(tup)
    return out

def to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list,)):
        return [to_jsonable(v) for v in obj]
    return obj

DEFAULT_HP = dict(
    property_csv_path = D("map_context_file/herb_properties.csv"),
    nes_csv_path      = D("map_context_file/herb_nes_matrix.csv"),
    mapping_csv_path  = D("map_context_file/herb_map_id.csv"),
    data_dir          = D("data"),
    results_dir       = D("results"),
    text_model        = D("map_context_file/bge-small-zh-v1.5"), 
    use_pretrained=False,
    emb_type='None',
    scorer_name="Dot",
    loss_type='Lambdarank',
    neg_sampling='popularity',
    num_negatives=100,
    lr=1e-3, weight_decay=1e-5,
    gat_in_dim=128, gat_hidden_dim=128, gat_layers=3,
    num_heads=4, attn_dropout=0.2,
    context_dim=256,
    residual_mode='off',
    ablation="none",
    num_epochs=50, patience=10,
    batch_size=32,
)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser(description="MAGED Train+Test (5-fold CV)")
    p.add_argument("--property_csv_path", type=str, default=DEFAULT_HP["property_csv_path"])
    p.add_argument("--nes_csv_path",      type=str, default=DEFAULT_HP["nes_csv_path"])
    p.add_argument("--mapping_csv_path",  type=str, default=DEFAULT_HP["mapping_csv_path"])
    p.add_argument("--data_dir",          type=str, default=DEFAULT_HP["data_dir"])
    p.add_argument("--results_dir",       type=str, default=DEFAULT_HP["results_dir"])
    p.add_argument("--text_model",        type=str, default=DEFAULT_HP["text_model"])
    p.add_argument("--output_dir", type=str, default=None, help="预训练节点向量目录（use_pretrained=1 时需要）")
    p.add_argument("--model_dir", type=str, default=None, help="保存模型的目录（默认 results_dir/model_results）")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--save_model", action="store_true", help="保存每折最佳模型与实体映射")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_pretrained", type=int, choices=[0,1], default=None)
    p.add_argument("--emb_type", type=str, default=None)
    p.add_argument("--scorer_name", type=str, default=None)
    p.add_argument("--loss_type", type=str, default=None)
    p.add_argument("--neg_sampling", type=str, default=None)
    p.add_argument("--num_negatives", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--gat_in_dim", type=int, default=None)
    p.add_argument("--gat_hidden_dim", type=int, default=None)
    p.add_argument("--gat_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--attn_dropout", type=float, default=None)
    p.add_argument("--context_dim", type=int, default=None)
    p.add_argument("--residual_mode", type=str, default=None)
    p.add_argument("--ablation", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    args = p.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    if args.model_dir is None:
        args.model_dir = str(Path(args.results_dir) / "model_results")
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    return args

def merge_hp_from_args(args) -> dict:
    H = dict(DEFAULT_HP)
    for k in DEFAULT_HP:
        if hasattr(args, k) and getattr(args, k) is not None:
            v = getattr(args, k)
            H[k] = bool(v) if (k == "use_pretrained") else v
    return H

def main():
    args = parse_args()
    set_seed(args.seed)
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    model_dir = args.model_dir or os.path.join(results_dir, "model_results")
    os.makedirs(model_dir, exist_ok=True)
    H = merge_hp_from_args(args)
    num_epochs, patience = H['num_epochs'], H['patience']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    AMP_DTYPE = (torch.bfloat16 if (device_type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16)
    scaler = GradScaler(device_type, enabled=(device_type == 'cuda' and AMP_DTYPE == torch.float16))
    ds, dn = build_text_nes_features(
        property_csv_path=args.property_csv_path,
        nes_csv_path=args.nes_csv_path,
        mapping_csv_path=args.mapping_csv_path,
        text_model=args.text_model,
        cache_dir=os.path.join(results_dir, "cache"),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    kg_builder = KnowledgeGraphBuilder(args.data_dir, herb_data_dict)
    kg_builder.load_all_triples()
    hetero_data, full_entity_id_map = kg_builder.build_hetero_graph()
    print(full_entity_id_map['TCM_ID'][:5])
    kg_builder.prepare_node_features(herb_data_dict)
    global_pos_dict = build_global_pos_dict(hetero_data)
    pretrained_embeddings = None
    if H.get('use_pretrained', True) and args.output_dir:
        pretrained_embeddings = load_node_embeddings(args.output_dir, emb_type=H['emb_type'])
    elif H.get('use_pretrained', True) and not args.output_dir:
        print("[WARN] use_pretrained=1 但未提供 --output_dir，将跳过加载预训练节点向量。")
    fusion_enc = FusionEncoder(text_dim=ds, nes_dim=dn, context_dim=H['context_dim'])
    model_init_params = dict(
        node_counts={nt: hetero_data[nt].num_nodes for nt in hetero_data.node_types},
        hidden_channels=H['gat_hidden_dim'],
        num_gnn_layers=H['gat_layers'],
        pretrained_embeddings=pretrained_embeddings,
        context_dim=H['context_dim'],
        num_heads=H['num_heads'],
        attn_dropout=H['attn_dropout'],
        in_dim=H['gat_in_dim'],
        fusion_encoder=fusion_enc,
        scorer_name=H.get('scorer_name', 'Concat_MLP'),
        scorer_kwargs=H.get('scorer_kwargs', None),
        residual_mode=H.get('residual_mode', 'inner+outer'),
        ablation=H.get('ablation', "none"),
    )
    kg_builder.split_datasets(n_splits=args.folds)
    all_test_metrics = []
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for fold in range(args.folds):
        print(f"\n========== Fold {fold+1}/{args.folds} ==========")
        t_train_idx, t_val_idx = kg_builder.target_folds[fold]
        target_train = kg_builder.t_remain[t_train_idx]
        target_val   = kg_builder.t_remain[t_val_idx]
        target_test  = kg_builder.t_test
        kg_builder._create_edge_masks(target_train, target_val, target_test)
        train_loader = get_train_loader(hetero_data, batch_size=H['batch_size'])
        val_loader   = get_val_loader(hetero_data,   batch_size=H['batch_size'])
        pop_w_target_global = (build_popularity_weights(hetero_data, split='train', device=device)
                               if H['neg_sampling'] == 'popularity' else None)
        model = MAGED(**model_init_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=H['lr'], weight_decay=H['weight_decay'])
        train_curve, val_curve = [], []
        best_val_loss = float('inf'); best_model_state = None; bad_count = 0
        pbar = tqdm(range(num_epochs), desc=f"Fold {fold+1} Training", unit="epoch")
        for epoch in pbar:
            model.train()
            total_target_loss, num_batches = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                entity_name_dict = batch_entity_names(batch, full_entity_id_map)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type, dtype=AMP_DTYPE, enabled=(device_type=='cuda')):
                    target_pred = model(batch, entity_name_dict)
                with autocast(device_type, enabled=False):
                    target_loss = compute_target_loss(
                        target_pred, batch,
                        num_neg_per_pos=H['num_negatives'],
                        loss_type=H['loss_type'],
                        sampling_strategy=H['neg_sampling'],
                        popularity_weight=pop_w_target_global,
                        global_pos_dict=global_pos_dict,
                    )
                if not torch.isfinite(target_loss):
                    continue
                loss32 = target_loss.float()
                if scaler.is_enabled():  # fp16
                    scaler.scale(loss32).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:                    # bf16 / cpu
                    loss32.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_target_loss += target_loss.item()
                num_batches += 1
            avg_train_target_loss = total_target_loss / max(1, num_batches)
            model.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    entity_name_dict = batch_entity_names(batch, full_entity_id_map)
                    with autocast(device_type, dtype=AMP_DTYPE, enabled=(device_type=='cuda')):
                        target_pred = model(batch, entity_name_dict)
                    with autocast(device_type, enabled=False):
                        target_loss = compute_target_loss(
                            target_pred, batch,
                            num_neg_per_pos=H['num_negatives'],
                            loss_type=H['loss_type'],
                            sampling_strategy=H['neg_sampling'],
                            popularity_weight=pop_w_target_global,
                            global_pos_dict=global_pos_dict,
                        )
                    if not torch.isfinite(target_loss):
                        continue
                    val_total += target_loss.item(); val_batches += 1
            avg_val_target_loss = val_total / max(1, val_batches)
            train_curve.append(avg_train_target_loss)
            val_curve.append(avg_val_target_loss)
            pbar.set_postfix({
                'train': f"{avg_train_target_loss:.4f}",
                'val':   f"{avg_val_target_loss:.4f}",
                'best':  f"{best_val_loss:.4f}" if np.isfinite(best_val_loss) else "inf",
                'bad':   bad_count
            })
            if avg_val_target_loss < best_val_loss:
                best_val_loss = avg_val_target_loss
                best_model_state = copy.deepcopy(model.state_dict())
                bad_count = 0
            else:
                bad_count += 1
            if bad_count >= H['patience']:
                tqdm.write(f"[Fold {fold+1}] Early stopping at epoch {epoch+1} (best val {best_val_loss:.4f})")
                break
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"[Train] Total elapsed: {elapsed:.3f}s")
        os.makedirs(model_dir, exist_ok=True)
        save_loss_plot(
            train_curve, val_curve,
            out_png=os.path.join(model_dir, "loss_curve_fold{fold+1}.png"),
            title=f"Fold {fold+1} Target Loss"
        )
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)
            if args.save_model:
                ckpt_path = os.path.join(model_dir, f"best_model_fold{fold+1}.pth")
                torch.save(best_model_state, ckpt_path)
                map_pth = os.path.join(model_dir, f"entity_map_fold{fold+1}.pth")
                torch.save(full_entity_id_map, map_pth)
                map_json = os.path.join(model_dir, f"entity_map_fold{fold+1}.json")
                try:
                    json.dump(to_jsonable(full_entity_id_map), open(map_json, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[WARN] 写 entity_map JSON 失败：{e}")
                try:
                    edge_types_from_state = extract_edge_types_from_state(best_model_state)
                except Exception:
                    edge_types_from_state = []
                meta = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "curve_name": None,
                    "fold": fold + 1,
                    "seed": int(args.seed),
                    "device": str(device),
                    "amp_dtype": str(AMP_DTYPE),
                    "hyperparams": to_jsonable({
                        "use_pretrained": H['use_pretrained'],
                        "emb_type": H['emb_type'],
                        "scorer_name": H['scorer_name'],
                        "loss_type": H['loss_type'],
                        "neg_sampling": H['neg_sampling'],
                        "num_negatives": H['num_negatives'],
                        "lr": H['lr'],
                        "weight_decay": H['weight_decay'],
                        "gat_in_dim": H['gat_in_dim'],
                        "gat_hidden_dim": H['gat_hidden_dim'],
                        "gat_layers": H['gat_layers'],
                        "num_heads": H['num_heads'],
                        "attn_dropout": H['attn_dropout'],
                        "context_dim": H['context_dim'],
                        "residual_mode": H['residual_mode'],
                        "ablation": H['ablation'],
                        "batch_size": H['batch_size'],
                        "num_epochs": H['num_epochs'],
                        "patience": H['patience'],
                        "text_model": args.text_model,
                    }),
                    "graph": {
                        "node_types": sorted(list(hetero_data.node_types)),
                        "node_counts": {nt: int(hetero_data[nt].num_nodes) for nt in hetero_data.node_types},
                        "edge_types_in_graph": to_jsonable(sorted(list(hetero_data.edge_types))),
                        "edge_types_from_state": to_jsonable(edge_types_from_state),
                    },
                    "paths": {
                        "checkpoint_pth": ckpt_path,
                        "entity_map_pth": map_pth,
                        "entity_map_json": map_json,
                    },
                    "version": {
                        "torch": torch.__version__,
                    },
                }
                meta_json = os.path.join(model_dir, f"model_meta_fold{fold+1}.json")
                try:
                    json.dump(meta, open(meta_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    print(f"[SAVE] model_meta saved: {meta_json}")
                except Exception as e:
                    print(f"[WARN] 写 model_meta JSON 失败：{e}")
        data_gpu = hetero_data.to(device)
        test_metrics = evaluate_model(model, data_gpu, full_entity_id_map, mask_type='test')
        all_test_metrics.append(test_metrics)
        print(f"[Fold{fold+1}] Test: HR@10={test_metrics.get('hr@10',0):.4f}, "
              f"NDCG@10={test_metrics.get('ndcg@10',0):.4f}, "
              f"Recall@100={test_metrics.get('recall@100',0):.4f}")
        del data_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()
    df = pd.DataFrame(all_test_metrics)
    print("\n五折测试集评估均值：");  print(df.mean(numeric_only=True))
    print("五折测试集评估标准差：");   print(df.std(numeric_only=True))
    out_csv = os.path.join(
        results_dir, "test_results_folds.csv"
    )
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Results saved to: {out_csv}")

if __name__ == "__main__":
    main()
