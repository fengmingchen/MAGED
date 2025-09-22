import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import io
import numpy as np
import torch
import torch.nn as nn
import random
from torch_geometric.loader import NeighborLoader
from itertools import cycle
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
TARGET_TYPES = ['Protein', 'TF', 'RBP']
HT_EDGE_NAME = 'herb_modulates_target'
SEED_NODE_TYPE = 'TCM_ID'

def batch_entity_names(batch, entity_name_map):
    entity_name_dict = {}
    for node_type in batch.node_types:
        if hasattr(batch[node_type], 'entity_id'):
            local_indices = batch[node_type].entity_id.cpu().tolist()
            names = []
            for i in local_indices:
                if i >= len(entity_name_map[node_type]):
                    raise IndexError(f"{node_type} 的局部索引 {i} 超出 entity_name_map 长度 {len(entity_name_map[node_type])}")
                names.append(entity_name_map[node_type][i])
            entity_name_dict[node_type] = names
    return entity_name_dict

def get_train_degree(hetero_data):
    num_herb = int(hetero_data['TCM_ID'].num_nodes)
    deg = torch.zeros(num_herb, dtype=torch.long, device='cpu')
    for tgt in TARGET_TYPES:
        et = (SEED_NODE_TYPE, HT_EDGE_NAME, tgt)
        if et in hetero_data.edge_index_dict and hasattr(hetero_data[et], 'train_mask'):
            mask = hetero_data[et].train_mask
            if mask is None:
                continue
            mask_cpu = mask.detach().cpu()
            eidx_cpu = hetero_data[et].edge_index.detach().cpu()
            h_ids = eidx_cpu[0, mask_cpu]
            deg.index_add_(0, h_ids, torch.ones_like(h_ids, device='cpu'))
    return deg 

def get_tcm_with_train_target(hetero_data):
    tcm_ids = set()
    for t in TARGET_TYPES:
        edge_type = (SEED_NODE_TYPE, HT_EDGE_NAME, t)
        if edge_type in hetero_data.edge_index_dict:
            mask = hetero_data[edge_type].train_mask
            if mask is None or mask.sum() == 0:
                continue
            h_ids = hetero_data[edge_type].edge_index[0, mask].detach().cpu()
            tcm_ids.update(h_ids.tolist())
    if not tcm_ids:
        raise ValueError("训练集中没有任何 TCM_ID 与 target 边")
    return torch.tensor(sorted(tcm_ids), dtype=torch.long)

def get_tcm_with_val_target(hetero_data):
    tcm_ids = set()
    for t in TARGET_TYPES:
        edge_type = (SEED_NODE_TYPE, HT_EDGE_NAME, t)
        if edge_type in hetero_data.edge_index_dict:
            mask = hetero_data[edge_type].val_mask
            if mask is None or mask.sum() == 0:
                continue
            h_ids = hetero_data[edge_type].edge_index[0, mask].detach().cpu()
            tcm_ids.update(h_ids.tolist())
    if not tcm_ids:
        raise ValueError("验证集中没有任何 TCM_ID 与 target 边")
    return torch.tensor(sorted(tcm_ids), dtype=torch.long)

def _interleave_by_degree(seeds, deg):
    if seeds.numel() == 0:
        return seeds
    sel_deg = deg[seeds]
    order = torch.argsort(sel_deg, stable=True)
    sorted_seeds = seeds[order].tolist()
    i, j = 0, len(sorted_seeds) - 1
    mixed = []
    while i <= j:
        mixed.append(sorted_seeds[i]); i += 1
        if i <= j:
            mixed.append(sorted_seeds[j]); j -= 1
    return torch.tensor(mixed, dtype=torch.long)

def get_train_loader(hetero_data, batch_size=16):
    input_tcm_ids = get_tcm_with_train_target(hetero_data)
    deg = get_train_degree(hetero_data)
    input_tcm_ids = input_tcm_ids[deg[input_tcm_ids] > 0]
    input_tcm_ids = _interleave_by_degree(input_tcm_ids, deg)
    num_neighbors_dict = {}
    for etype in hetero_data.edge_types:
        if etype[1] == HT_EDGE_NAME:
                num_neighbors_dict[etype] = [-1, 16, 8, 4]
        else:
            num_neighbors_dict[etype] = [16, 8, 8, 4]
    return NeighborLoader(
        hetero_data,
        input_nodes=(SEED_NODE_TYPE, input_tcm_ids),
        num_neighbors=num_neighbors_dict,
        batch_size=batch_size,
        shuffle=False
    )

def get_val_loader(hetero_data, batch_size=16):
    input_tcm_ids = get_tcm_with_val_target(hetero_data)  
    num_neighbors_dict = {}
    for etype in hetero_data.edge_types:
        if etype[1] == HT_EDGE_NAME:
            num_neighbors_dict[etype] = [-1, 16, 8, 4]
        else:
            num_neighbors_dict[etype] = [16, 8, 8, 4]
    return NeighborLoader(
        hetero_data,
        input_nodes=(SEED_NODE_TYPE, input_tcm_ids),
        num_neighbors=num_neighbors_dict,
        batch_size=batch_size,
        shuffle=False
    )

def load_node_embeddings(embedding_dir, emb_type='DeepWalk'):
    entity2id = {}
    with open(f"{embedding_dir}/entity2id.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            eid = " ".join(parts[:-1])
            idx = parts[-1]
            entity2id[eid] = int(idx)
    def _read_embedding_file(path: str) -> np.ndarray:
        if path.endswith(".npy"):
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float32)
            return arr
        if path.endswith(".npz"):
            z = np.load(path)
            key = z.files[0]
            arr = np.asarray(z[key], dtype=np.float32)
            return arr
        try:
            arr = np.loadtxt(path, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr.astype(np.float32)
        except Exception:
            pass
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        txt = txt.replace('(', '').replace(')', '')
        try:
            z = np.loadtxt(io.StringIO(txt), dtype=np.complex128)
            if z.ndim == 1:
                z = z.reshape(1, -1)
            arr = np.hstack([z.real, z.imag]).astype(np.float32)
            return arr
        except Exception as e:
            raise ValueError(f"无法解析预训练向量文件: {path}；"
                             f"既不是纯实数，也不是标准复数文本。原始错误: {e}")
    emb_file_map = {
        'TCM_ID': f'{emb_type}_embedding.txt',
        'Protein': f'{emb_type}_embedding.txt',
        'TF': f'{emb_type}_embedding.txt',
        'RBP': f'{emb_type}_embedding.txt',
        'mRNA': f'{emb_type}_embedding.txt',
        'TCM_symptom_ID': f'{emb_type}_embedding.txt',
        'UMLS_id': f'{emb_type}_embedding.txt',
        'lncRNA': f'{emb_type}_embedding.txt',
        'miRNA': f'{emb_type}_embedding.txt',
        'DNA': f'{emb_type}_embedding.txt'
    }
    emb_dict = {}
    for node_type, file_name in emb_file_map.items():
        emb_path = f"{embedding_dir}/{file_name}"
        if not os.path.exists(emb_path):
            continue
        arr = _read_embedding_file(emb_path)
        emb_tensor = torch.tensor(arr, dtype=torch.float32)
        emb_dict[node_type] = (emb_tensor, entity2id)
    return emb_dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bpr_pairwise_loss(pos_scores, neg_scores):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=(pos_scores.device if len(pos_scores) > 0 else neg_scores.device), requires_grad=True)
    pos = pos_scores.view(-1, 1)
    neg = neg_scores.view(1, -1)
    diff = pos - neg
    loss = -F.logsigmoid(diff)
    return loss.mean()

def margin_pairwise_loss(pos_scores, neg_scores, margin=1.0):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=(pos_scores.device if len(pos_scores) > 0 else neg_scores.device), requires_grad=True)
    pos = pos_scores.view(-1, 1)
    neg = neg_scores.view(1, -1)
    diff = pos - neg
    loss = F.relu(margin - diff)
    return loss.mean()

def bce_loss(pos_scores, neg_scores):
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    return F.binary_cross_entropy_with_logits(scores, labels)

def softmax_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, tau: float = 1.0):
    if pos_scores.numel() == 0:
        device = neg_scores.device if neg_scores.numel() > 0 else 'cpu'
        return torch.tensor(0.0, device=device)
    all_scores = torch.cat([pos_scores, neg_scores], dim=0) if neg_scores.numel() > 0 else pos_scores
    all_scores = all_scores / tau
    pos_scores = pos_scores / tau
    loss = torch.logsumexp(all_scores, dim=0) - torch.logsumexp(pos_scores, dim=0)
    loss = loss / max(1, pos_scores.numel())
    return loss

def delta_ndcg(pos_rank, neg_rank, num_items):
    def dcg(rank):
        return 1.0 / np.log2(rank + 2)
    ideal_dcg = sum([dcg(r) for r in range(num_items)])
    return abs(dcg(pos_rank) - dcg(neg_rank)) / ideal_dcg if ideal_dcg > 0 else 0.0

def lambdarank_ndcg_approx_loss(pos_scores, neg_scores, pos_gain=1, neg_gain=0):
    P, N = len(pos_scores), len(neg_scores)
    if P == 0 or N == 0:
        device = pos_scores.device if P > 0 else neg_scores.device
        return torch.tensor(0.0, device=device, requires_grad=True)
    pos_ranks = torch.arange(P, device=pos_scores.device)
    neg_ranks = torch.arange(N, device=neg_scores.device) + P
    idcg = torch.sum((2 ** pos_gain - 1) / torch.log2(torch.arange(2, 2 + P, device=pos_scores.device).float()))
    dcg_pos = (2 ** pos_gain - 1) / torch.log2(pos_ranks.view(-1, 1) + 2)  # [P,1]
    dcg_neg = (2 ** neg_gain - 1) / torch.log2(neg_ranks.view(1, -1) + 2)  # [1,N]
    delta_ndcg = torch.abs(dcg_pos - dcg_neg) / (idcg + 1e-10)  # [P, N]
    diff = pos_scores.view(-1, 1) - neg_scores.view(1, -1)  # [P, N]
    loss = delta_ndcg * F.softplus(-diff)
    return loss.mean()

def auto_expand_feature(x, target_dim):
    if x.size(-1) == target_dim:
        return x
    lin = nn.Linear(x.size(-1), target_dim).to(x.device)
    nn.init.xavier_uniform_(lin.weight)
    with torch.no_grad():
        x_new = lin(x)
    return x_new

def _safe_mask(storage, split: str):
    mname = f"{split}_mask"
    if hasattr(storage, mname):
        m = getattr(storage, mname)
        if m is not None:
            return m
    return torch.ones(storage.edge_index.size(1), dtype=torch.bool, device=storage.edge_index.device)

@torch.no_grad()
def build_popularity_weights(hetero_data, split: str = 'train', device=None):
    if device is None:
        device = next(iter(hetero_data.node_types and [torch.device('cpu')] or [torch.device('cpu')]))
    pop_w_target = {}
    for ttype in TARGET_TYPES:
        et = ('TCM_ID', 'herb_modulates_target', ttype)
        if et not in hetero_data.edge_types:
            pop_w_target[ttype] = torch.zeros(0, dtype=torch.float32, device=device)
            continue
        num_dst = hetero_data[ttype].num_nodes
        ei = hetero_data[et].edge_index
        mask = _safe_mask(hetero_data[et], split)
        ei = ei[:, mask]
        deg = torch.zeros(num_dst, dtype=torch.float32, device=device)
        if ei.numel() > 0:
            tgt = ei[1].to(device)
            one = torch.ones_like(tgt, dtype=torch.float32, device=device)
            deg.index_add_(0, tgt, one)
        s = deg.sum()
        if s <= 0:
            prob = torch.full((num_dst,), 1.0 / max(num_dst, 1), dtype=torch.float32, device=device) if num_dst > 0 else deg
        else:
            prob = deg / s
        pop_w_target[ttype] = prob
    return pop_w_target

def build_global_pos_dict(hetero_data):
    out = {k: defaultdict(set) for k in TARGET_TYPES}
    rel = 'herb_modulates_target'
    def node_gids(nt):
        if hasattr(hetero_data[nt], 'entity_id') and hetero_data[nt].entity_id is not None:
            return hetero_data[nt].entity_id                 # Tensor[num_nodes]
        return torch.arange(hetero_data[nt].num_nodes, dtype=torch.long)
    g_h_all = node_gids('TCM_ID')
    for tgt in TARGET_TYPES:
        et = ('TCM_ID', rel, tgt)
        if et not in hetero_data.edge_index_dict:
            continue
        ei = hetero_data[et].edge_index                     # [2, E]
        g_t_all = node_gids(tgt)
        gh = g_h_all[ei[0]].tolist()
        gt = g_t_all[ei[1]].tolist()
        for h_id, t_id in zip(gh, gt):
            out[tgt][int(h_id)].add(int(t_id))
    return out

@torch.no_grad()
def export_all_embeddings_csv(model, hetero_data, full_entity_id_map, out_csv_path):
    device = next(model.parameters()).device
    model.eval()
    data = hetero_data.to(device)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    node_types = list(data.node_types)
    prefer_order = []
    if 'TCM_ID' in node_types:
        prefer_order.append('TCM_ID')
    prefer_order.extend([nt for nt in node_types if nt != 'TCM_ID'])
    name_dict = {}
    for nt in node_types:
        if (full_entity_id_map is not None) and (nt in full_entity_id_map):
            name_dict[nt] = list(full_entity_id_map[nt])
        else:
            name_dict[nt] = [f"{nt}_{i}" for i in range(data[nt].num_nodes)]
    x_dict = {}
    with torch.inference_mode():
        for nt in node_types:
            n = data[nt].num_nodes
            if n <= 0:
                continue
            try:
                x = model._lookup_embed(nt, name_dict.get(nt, []), data, device)
            except AttributeError:
                raise RuntimeError("model._lookup_embed(nt, names, data, device) 未定义，请保持与现有实现一致。")
            if x.size(0) < n:
                pad = torch.zeros(n - x.size(0), x.size(1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            elif x.size(0) > n:
                x = x[:n]
            x_dict[nt] = x
        try:
            c_h = model._get_context(data, device) if 'TCM_ID' in x_dict else None
        except AttributeError:
            c_h = None
        if ('TCM_ID' in x_dict) and (c_h is not None):
            tcm = x_dict['TCM_ID']
            n = min(tcm.size(0), c_h.size(0))
            fused = model.tcm_fusion(torch.cat([tcm[:n], c_h[:n]], dim=-1))
            if n < tcm.size(0):
                pad = torch.zeros(tcm.size(0)-n, fused.size(1), device=device, dtype=fused.dtype)
                fused = torch.cat([fused, pad], dim=0)
            x_dict['TCM_ID'] = fused
        for nt in list(x_dict.keys()):
            if nt == 'TCM_ID':
                continue
            try:
                x_dict[nt] = model.pre_gnn_proj(x_dict[nt])
            except AttributeError:
                pass
        context_src_dict, context_dst_dict = {}, {}
        if c_h is not None:
            for et, eidx in data.edge_index_dict.items():
                src_t, _, dst_t = et
                if src_t == 'TCM_ID':
                    context_src_dict[et] = c_h
                if dst_t == 'TCM_ID':
                    context_dst_dict[et] = c_h
        try:
            out = model.gnn(
                x_dict=x_dict,
                edge_index_dict=data.edge_index_dict,
                node_counts={nt: data[nt].num_nodes for nt in node_types},
                context_src_dict=context_src_dict,
                context_dst_dict=context_dst_dict,
            )
        except TypeError:
            out = model.gnn(x_dict, data.edge_index_dict)
        for nt, xin in x_dict.items():
            if nt not in out:
                out[nt] = xin
            n = data[nt].num_nodes
            if out[nt].size(0) < n:
                pad = torch.zeros(n - out[nt].size(0), out[nt].size(1), device=out[nt].device, dtype=out[nt].dtype)
                out[nt] = torch.cat([out[nt], pad], dim=0)
            elif out[nt].size(0) > n:
                out[nt] = out[nt][:n]
        blocks = []
        per_type_paths = []
        for nt in prefer_order:
            if nt not in out:
                continue
            E = out[nt].detach().cpu().float().numpy()
            names = name_dict[nt][:E.shape[0]]
            df = pd.DataFrame(E)
            df.insert(0, 'name', names)
            df.insert(1, 'type', nt)
            blocks.append(df)
        if not blocks:
            print("[WARN] 没有可导出的类型。")
            return
        big = pd.concat(blocks, ignore_index=True)
        big.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
        print(f"[EXPORT] all embeddings -> {out_csv_path}  shape={big.shape}")
        try:
            from utils_embedding import TARGET_TYPES
            target_types = [t for t in TARGET_TYPES if t in node_types]
        except Exception:
            target_types = [nt for nt in node_types if nt != 'TCM_ID']
        subset_types = (['TCM_ID'] if 'TCM_ID' in node_types else []) + target_types
        herb_target_df = big[big['type'].isin(subset_types)].reset_index(drop=True)
        sub_path = out_csv_path.replace('.csv', '__herb_target.csv')
        herb_target_df.to_csv(sub_path, index=False, encoding='utf-8-sig')
        print(f"[EXPORT] herb+target subset -> {sub_path}  shape={herb_target_df.shape}")


