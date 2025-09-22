import os, gc, json, glob, random, re, copy
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .kgbuilder import KnowledgeGraphBuilder
from .encoder import build_text_nes_features, herb_data_dict
from .maged import MAGED, evaluate_model, FusionEncoder
from .utils_embedding import (
    batch_entity_names, build_global_pos_dict, TARGET_TYPES,
)
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
def D(*parts: str) -> str:
    return str((REPO_ROOT / Path(*parts)).resolve())
INFER_DEFAULTS = dict(
    property_csv_path = D("map_context_file/herb_properties.csv"),
    nes_csv_path      = D("map_context_file/herb_nes_matrix.csv"),
    mapping_csv_path  = D("map_context_file/herb_map_id.csv"),
    data_dir          = D("data"),
    results_dir       = D("results"),
    text_model        = D("map_context_file/bge-small-zh-v1.5"),
    model_dir         = D("results/model_results"),
)

def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

STRICT_EDGE_KEY_RE = re.compile(r"gnn\.convs\.\d+\.convs\.<([^>]+)>\.")
LOOSE_TAG_RE = re.compile(r"<([^>]+)>")

def extract_edge_types_from_state(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
    seen, out = set(), []
    for k in state_dict.keys():
        m = STRICT_EDGE_KEY_RE.search(k)
        if m:
            parts = m.group(1).split("___")
            if len(parts) == 3:
                tup = (parts[0], parts[1], parts[2])
                if tup not in seen:
                    seen.add(tup); out.append(tup)
    if len(out) > 0:
        return out
    for k in state_dict.keys():
        for m in LOOSE_TAG_RE.finditer(k):
            parts = m.group(1).split("___")
            if len(parts) == 3:
                tup = (parts[0], parts[1], parts[2])
                if tup not in seen:
                    seen.add(tup); out.append(tup)
    return out

def _edge_complete_in_state(edge: Tuple[str,str,str], state: Dict[str, torch.Tensor], num_layers: int) -> bool:
    tag_esc = re.escape(f"<{edge[0]}___{edge[1]}___{edge[2]}>")
    for l in range(int(num_layers)):
        pat_src = re.compile(rf"gnn\.convs\.{l}\.convs\.{tag_esc}\.lin_src\.weight")
        pat_dst = re.compile(rf"gnn\.convs\.{l}\.convs\.{tag_esc}\.lin_dst\.weight")
        if not any(pat_src.search(k) for k in state.keys()):
            return False
        if not any(pat_dst.search(k) for k in state.keys()):
            return False
    return True

def _find_meta(model_dir: str, fold: int) -> str:
    metas = glob.glob(os.path.join(model_dir, f"*model_meta_fold{fold}.json"))
    if not metas: return ""
    metas.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return metas[0]

def _load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _override_hparams_from_meta(defaults: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    hp = dict(defaults); m = meta.get("hyperparams", {}) or {}
    keys = ["gat_in_dim","gat_hidden_dim","gat_layers","num_heads","attn_dropout",
            "context_dim","residual_mode","ablation","scorer_name","scorer_kwargs"]
    for k in keys:
        if k in m and m[k] is not None:
            hp[k] = m[k]
    return hp

def _reconcile_heterodata_edge_types(hetero_data, desired_edge_types: List[Tuple[str, str, str]]):
    for et in list(hetero_data.edge_types):
        if et not in desired_edge_types:
            try:
                del hetero_data[et]
                print(f"[reconcile] removed extra edge_type: {et}")
            except Exception as e:
                print(f"[reconcile][warn] failed to remove {et}: {e}")
    for et in desired_edge_types:
        if et not in hetero_data.edge_types:
            try:
                hetero_data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                hetero_data[et].train_mask = torch.zeros((0,), dtype=torch.bool)
                hetero_data[et].val_mask   = torch.zeros((0,), dtype=torch.bool)
                hetero_data[et].test_mask  = torch.zeros((0,), dtype=torch.bool)
                print(f"[reconcile] added missing edge_type (empty): {et}")
            except Exception as e:
                print(f"[reconcile][warn] failed to add {et}: {e}")

def _make_warmup_graph(hetero_data, edges: List[Tuple[str,str,str]]):
    warm = copy.deepcopy(hetero_data)
    for (src, rel, dst) in edges:
        if src not in warm.node_types or dst not in warm.node_types:
            continue
        src_n = int(getattr(warm[src], "num_nodes", 0))
        dst_n = int(getattr(warm[dst], "num_nodes", 0))
        if src_n == 0 or dst_n == 0:
            continue
        need_place = False
        if (src, rel, dst) not in warm.edge_types:
            need_place = True
        else:
            ei = getattr(warm[(src,rel,dst)], "edge_index", None)
            if ei is None or ei.numel() == 0:
                need_place = True
        if need_place:
            warm[(src,rel,dst)].edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            warm[(src,rel,dst)].train_mask = torch.zeros((1,), dtype=torch.bool)
            warm[(src,rel,dst)].val_mask   = torch.zeros((1,), dtype=torch.bool)
            warm[(src,rel,dst)].test_mask  = torch.zeros((1,), dtype=torch.bool)
    return warm

def parse_args():
    p = argparse.ArgumentParser(
        description="MAGED Inference: 直接加载已训练好的折别模型进行测试评估与Top-K候选导出（无训练）"
    )
    p.add_argument("--property_csv_path", type=str, default=INFER_DEFAULTS["property_csv_path"],
                   help="中药属性 CSV")
    p.add_argument("--nes_csv_path", type=str, default=INFER_DEFAULTS["nes_csv_path"],
                   help="NES 矩阵 CSV")
    p.add_argument("--mapping_csv_path", type=str, default=INFER_DEFAULTS["mapping_csv_path"],
                   help="中药中文名↔TCM_ID 映射表 CSV")
    p.add_argument("--data_dir", type=str, default=INFER_DEFAULTS["data_dir"],
                   help="知识图谱三元组目录")
    p.add_argument("--results_dir", type=str, default=INFER_DEFAULTS["results_dir"],
                   help="推理输出目录（指标与TopK）")
    p.add_argument("--text_model", type=str, default=INFER_DEFAULTS["text_model"],
                   help="本地文本编码模型目录（如 bge-small-zh-v1.5）")
    p.add_argument("--model_dir", type=str, default=INFER_DEFAULTS["model_dir"],
                   help="保存“每折最佳模型(.pth)”与实体映射(.pth/.json)和 model_meta 的目录")
    p.add_argument("--folds", type=int, default=5, help="需要评测的折数（与训练时保持一致）")
    p.add_argument("--model_glob", type=str, default=None,
                   help="可选：自定义匹配模式寻找每折权重，如 '*best_model_fold{fold}.pth'；默认自动匹配")
    p.add_argument("--gat_in_dim", type=int, default=128)
    p.add_argument("--gat_hidden_dim", type=int, default=128)
    p.add_argument("--gat_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--context_dim", type=int, default=256)
    p.add_argument("--residual_mode", type=str, default="off")
    p.add_argument("--ablation", type=str, default="none")
    p.add_argument("--scorer_name", type=str, default="Dot")
    p.add_argument("--topk_herb", type=str, default=None,
                   help='指定中药 ID 或中文名（如 "TCM_ID:HTHP00105" 或 "黄芩"）；为空则不导出Top-K')
    p.add_argument("--topk_k", type=int, default=300, help="Top-K 的 K 值")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = p.parse_args()
    return args

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _find_checkpoint(model_dir: str, fold: int, model_glob: str = None) -> str:
    if model_glob:
        pattern = model_glob.format(fold=fold)
        paths = glob.glob(os.path.join(model_dir, pattern))
    else:
        paths = glob.glob(os.path.join(model_dir, f"*best_model_fold{fold}.pth"))
    assert len(paths) > 0, f"[infer] 找不到第 {fold} 折模型权重，请检查 --model_dir 或 --model_glob。"
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def _find_entity_map(model_dir: str, fold: int) -> str:
    pths = glob.glob(os.path.join(model_dir, f"*entity_map_fold{fold}.pth"))
    jsns = glob.glob(os.path.join(model_dir, f"*entity_map_fold{fold}.json"))
    if pths: return pths[0]
    if jsns: return jsns[0]
    return ""

def _assert_same_mapping(built_map: Dict[str, List[str]], saved_map: Dict[str, List[str]]):
    for nt in saved_map.keys():
        assert nt in built_map, f"[infer] 训练时包含节点类型 {nt}，但当前构图缺失。"
        a, b = list(map(str, saved_map[nt])), list(map(str, built_map[nt]))
        assert len(a) == len(b), f"[infer] 节点数不一致: {nt} 训练={len(a)} 推理={len(b)}"
        assert all(x == y for x, y in zip(a, b)), (
            f"[infer] 节点次序与训练不一致: {nt}。请确保构图与训练时完全一致或复用训练时的 entity_id_map。"
        )

def _maybe_map_herb_name_to_id(mapping_csv_path: str, herb_query: str) -> str:
    if herb_query.startswith("TCM_ID:"):
        return herb_query
    if not os.path.exists(mapping_csv_path):
        return herb_query
    try:
        mdf = pd.read_csv(mapping_csv_path)
        id_col = "TCM_ID" if "TCM_ID" in mdf.columns else None
        name_cols = [c for c in mdf.columns if any(k in c.lower() for k in ["name", "中药", "herb"])]
        if id_col and name_cols:
            rows = mdf[[id_col] + name_cols]
            hit = rows[rows.apply(lambda r: any(str(herb_query) == str(r[c]) for c in name_cols), axis=1)]
            if len(hit) > 0:
                return str(hit.iloc[0][id_col])
    except Exception:
        pass
    return herb_query

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if (args.device is None and torch.cuda.is_available()) else (args.device or 'cpu'))
    os.makedirs(args.results_dir, exist_ok=True)
    ds, dn = build_text_nes_features(
        property_csv_path=args.property_csv_path,
        nes_csv_path=args.nes_csv_path,
        mapping_csv_path=args.mapping_csv_path,
        text_model=args.text_model,
        cache_dir=os.path.join(args.results_dir, "cache"),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    kg_builder = KnowledgeGraphBuilder(args.data_dir, herb_data_dict)
    kg_builder.load_all_triples()
    hetero_data, full_entity_id_map = kg_builder.build_hetero_graph()
    kg_builder.prepare_node_features(herb_data_dict)
    _ = build_global_pos_dict(hetero_data)
    kg_builder.split_datasets(n_splits=args.folds)
    all_test_metrics = []
    for fold in range(1, args.folds + 1):
        print(f"\n========== Inference Fold {fold}/{args.folds} ==========")
        meta_path = _find_meta(args.model_dir, fold)
        meta = {}
        if meta_path and os.path.exists(meta_path):
            try:
                meta = _load_meta(meta_path)
                print(f"[meta] loaded: {meta_path}")
            except Exception as e:
                print(f"[meta][warn] failed to load meta: {e}")
        ckpt_path = _find_checkpoint(args.model_dir, fold, args.model_glob)
        state = safe_torch_load(ckpt_path, map_location="cpu")
        real_state = state["state_dict"] if isinstance(state, dict) and isinstance(state.get("state_dict"), dict) else state
        trained_edges = extract_edge_types_from_state(real_state)
        if len(trained_edges) == 0:
            raise RuntimeError("[infer] 无法从 checkpoint 的 state_dict 键名中解析出任何边类型。"
                               "请检查保存键名或正则。")
        hp_defaults = dict(
            gat_in_dim=args.gat_in_dim, gat_hidden_dim=args.gat_hidden_dim, gat_layers=args.gat_layers,
            num_heads=args.num_heads, attn_dropout=args.attn_dropout, context_dim=args.context_dim,
            residual_mode=args.residual_mode, ablation=args.ablation, scorer_name=args.scorer_name,
            scorer_kwargs=None,
        )
        hp_used = _override_hparams_from_meta(hp_defaults, meta) if meta else hp_defaults
        complete_edges = [et for et in trained_edges if _edge_complete_in_state(et, real_state, hp_used["gat_layers"])]
        if len(complete_edges) == 0:
            raise RuntimeError("[infer] 从 checkpoint 解析到的关系里，没有任何一条在所有 GNN 层都具备 lin_src/lin_dst 权重；无法严格加载。")
        target_edges = complete_edges
        print(f"[edges] from_ckpt={len(trained_edges)}  complete_used={len(complete_edges)}")
        _reconcile_heterodata_edge_types(hetero_data, target_edges)
        t_train_idx, t_val_idx = kg_builder.target_folds[fold - 1]
        target_train = kg_builder.t_remain[t_train_idx]
        target_val   = kg_builder.t_remain[t_val_idx]
        target_test  = kg_builder.t_test
        kg_builder._create_edge_masks(target_train, target_val, target_test)
        ent_map_path = ""
        if meta:
            ent_map_path = meta.get("paths", {}).get("entity_map_pth", "") or meta.get("paths", {}).get("entity_map_json", "") or ""
        if not ent_map_path:
            ent_map_path = _find_entity_map(args.model_dir, fold)
        if ent_map_path.endswith(".pth") and os.path.exists(ent_map_path):
            saved_map = safe_torch_load(ent_map_path, map_location="cpu")
            _assert_same_mapping(full_entity_id_map, saved_map)
        elif ent_map_path.endswith(".json") and os.path.exists(ent_map_path):
            with open(ent_map_path, "r", encoding="utf-8") as f:
                saved_map = json.load(f)
            _assert_same_mapping(full_entity_id_map, saved_map)
        else:
            print("[warn] 未找到保存的 entity_map；默认认为当前构图与训练一致。")
        print("[hparams] used:", {k: hp_used[k] for k in ["gat_in_dim","gat_hidden_dim","gat_layers","num_heads","attn_dropout","context_dim","residual_mode","ablation","scorer_name"]})
        fusion_enc = FusionEncoder(text_dim=ds, nes_dim=dn, context_dim=hp_used["context_dim"])
        model_init_params = dict(
            node_counts={nt: hetero_data[nt].num_nodes for nt in hetero_data.node_types},
            hidden_channels=hp_used["gat_hidden_dim"],
            num_gnn_layers=hp_used["gat_layers"],
            pretrained_embeddings=None,
            context_dim=hp_used["context_dim"],
            num_heads=hp_used["num_heads"],
            attn_dropout=hp_used["attn_dropout"],
            in_dim=hp_used["gat_in_dim"],
            fusion_encoder=fusion_enc,
            scorer_name=hp_used["scorer_name"],
            scorer_kwargs=hp_used.get("scorer_kwargs", None),
            residual_mode=hp_used["residual_mode"],
            ablation=hp_used["ablation"],
        )
        model = MAGED(**model_init_params).to(device)
        model.eval()
        warm_graph = _make_warmup_graph(hetero_data, target_edges)
        with torch.no_grad():
            entity_name_dict_warm = batch_entity_names(warm_graph, full_entity_id_map)
            _ = model(warm_graph.to(device), entity_name_dict_warm)
        del warm_graph
        try:
            model.load_state_dict(real_state, strict=True)
            print("[infer] strict load OK (after warmup).")
        except RuntimeError as e:
            raise RuntimeError(f"[infer] strict load failed after warmup: {e}")
        print(f"[infer] loaded checkpoint: {ckpt_path}")
        data_gpu = hetero_data.to(device)
        test_metrics = evaluate_model(model, data_gpu, full_entity_id_map, mask_type='test')
        all_test_metrics.append(test_metrics)
        print(f"[Fold{fold}] Test Target Metrics: "
              f"HR@10={test_metrics.get('hr@10',0):.4f}, "
              f"NDCG@10={test_metrics.get('ndcg@10',0):.4f}, "
              f"Recall@100={test_metrics.get('recall@100',0):.4f}")
        if args.topk_herb:
            with torch.no_grad():
                entity_name_dict = batch_entity_names(data_gpu, full_entity_id_map)
                target_pred_full = model(data_gpu, entity_name_dict)
            herb_query = _maybe_map_herb_name_to_id(args.mapping_csv_path, str(args.topk_herb).strip())
            tcm_list = list(map(str, full_entity_id_map['TCM_ID']))
            herb2local = {name: i for i, name in enumerate(tcm_list)}
            assert herb_query in herb2local, f"[infer] 找不到中药：{args.topk_herb}（既非 TCM_ID 也非映射表可解析的中文名）"
            h_idx = herb2local[herb_query]
            size_map = {t: (data_gpu[t].num_nodes if t in data_gpu.node_types else 0) for t in TARGET_TYPES}
            offsets, _ptr = {}, 0
            for t in TARGET_TYPES:
                offsets[t] = _ptr; _ptr += size_map.get(t, 0)
            scores = target_pred_full[h_idx].detach().float().cpu()
            exclude_cols = []
            for t in TARGET_TYPES:
                et = ('TCM_ID', 'herb_modulates_target', t)
                if et in data_gpu.edge_types and hasattr(data_gpu[et], 'train_mask') and getattr(data_gpu[et], 'train_mask') is not None:
                    ei = data_gpu[et].edge_index.detach().cpu()
                    m  = data_gpu[et].train_mask.detach().cpu().bool()
                    if ei.numel() == 0 or m.numel() == 0: continue
                    ei_tr = ei[:, m]
                    sel = (ei_tr[0] == h_idx).nonzero(as_tuple=True)[0]
                    if sel.numel() == 0: continue
                    tgt_local = ei_tr[1, sel]
                    cols = tgt_local + offsets[t]
                    exclude_cols.append(cols)
            if len(exclude_cols) > 0:
                exclude_cols = torch.unique(torch.cat(exclude_cols))
                scores_filtered = scores.clone(); scores_filtered[exclude_cols] = float('-inf')
            else:
                scores_filtered = scores
            K = int(args.topk_k)
            top_idx = torch.argsort(scores_filtered, descending=True)[:K].tolist()
            out_rows = []
            for j in top_idx:
                for t in TARGET_TYPES:
                    start = offsets[t]; end = start + size_map.get(t, 0)
                    if start <= j < end:
                        local_j = j - start
                        name = full_entity_id_map[t][local_j]
                        out_rows.append({'rank': len(out_rows)+1, 'type': t, 'target': name, 'score': float(scores[j])})
                        break
            df_top = pd.DataFrame(out_rows)
            out_dir = os.path.join(args.results_dir, "topk"); os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"top{K}_{herb_query.replace(':','_')}_fold{fold}.csv")
            df_top.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[TopK] Herb={herb_query} 的前 {K} 个候选（预览前20行）：")
            print(df_top.head(20).to_string(index=False))
            print(f"[TopK] 已保存至：{csv_path}")
        del model, data_gpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    df = pd.DataFrame(all_test_metrics)
    print("\n[Infer] 五折测试集评估均值：")
    print(df.mean(numeric_only=True))
    print("[Infer] 五折测试集评估标准差：")
    print(df.std(numeric_only=True))
    out_csv = os.path.join(args.results_dir, "infer_test_results_folds.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Inference results saved to: {out_csv}")

if __name__ == "__main__":
    main()
