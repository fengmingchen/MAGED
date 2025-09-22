import os
# DEBUG_MAGED = os.environ.get("DEBUG_MAGED", "0") == "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_scatter import scatter_add,scatter
from .kgbuilder import *
from .encoder import *
from .utils_embedding import *
from .utils_embedding import TARGET_TYPES
from .scorers import *
from torch_geometric.utils import negative_sampling, softmax
import matplotlib.pyplot as plt
from contextlib import nullcontext
class ContextAwareGATConv(nn.Module):
    def __init__(self, out_channels, context_dim, heads=4, dropout=0.2, use_inner_residual=True):
        super().__init__()
        self.heads = heads
        self.context_dim = context_dim
        self.out_channels = out_channels
        self.use_inner_residual = use_inner_residual
        self.lin_src = None
        self.lin_dst = None
        self.residual = None
        self.residual_adjust = None
        self.context_att = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ELU(),
            nn.Linear(context_dim, heads),
            nn.Sigmoid()
        )
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(heads * out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def build_layers(self, in_channels_src, in_channels_dst):
        self.lin_src = nn.Linear(in_channels_src, self.heads * self.out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels_dst, self.heads * self.out_channels, bias=False)
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        if self.use_inner_residual:
            self.residual = nn.Sequential(
                nn.Linear(in_channels_dst, self.heads * self.out_channels),
                nn.ReLU()
            )
            nn.init.xavier_uniform_(self.residual[0].weight)
            if self.residual[0].bias is not None:
                nn.init.zeros_(self.residual[0].bias)

    def forward(self, x, edge_index, **kwargs):
        x_src_orig, x_dst_orig = x
        row, col = edge_index[0], edge_index[1]
        context_src = kwargs.get('context_src', kwargs.get('context', None))
        context_dst = kwargs.get('context_dst', None)
        if self.lin_src is None:
            self.build_layers(x_src_orig.size(1), x_dst_orig.size(1))
            self.to(x_src_orig.device)
        x_src_proj = self.lin_src(x_src_orig).view(-1, self.heads, self.out_channels)
        x_dst_proj = self.lin_dst(x_dst_orig).view(-1, self.heads, self.out_channels)
        x_src_edge = x_src_proj.index_select(0, row)   # [E, H, C]
        x_dst_edge = x_dst_proj.index_select(0, col)   # [E, H, C]
        num_dst_nodes = x_dst_proj.size(0)
        alpha = self.compute_attention(
            x_src_edge, x_dst_edge, row, col,
            context_src, context_dst, num_dst_nodes,
            device=x_src_orig.device
        )
        out = self.propagate(edge_index, x=x_src_edge, alpha=alpha,
                             size=(x_src_orig.size(0), x_dst_orig.size(0)))
        out = out.view(-1, self.heads * self.out_channels)
        if self.residual is not None:
            residual = self.residual(x_dst_orig)
            if residual.size(-1) != out.size(-1):
                if self.residual_adjust is None:
                    self.residual_adjust = nn.Linear(residual.size(-1), out.size(-1)).to(out.device)
                residual = self.residual_adjust(residual)
            out = out + residual

        return self.norm(self.dropout(out))

    def compute_attention(self, x_src, x_dst, row, col,
                          context_src, context_dst, num_dst_nodes, device):
        alpha = (x_src * self.att_src).sum(dim=-1) + (x_dst * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)  # [E, H]
        if context_src is not None and row.numel() > 0:
            ctx = context_src.to(device)
            row_c = torch.clamp(row, 0, ctx.size(0) - 1)
            m_src = self.context_att(ctx.index_select(0, row_c))  # [E, H]
            alpha = alpha * m_src
        if context_dst is not None and col.numel() > 0:
            ctx = context_dst.to(device)
            col_c = torch.clamp(col, 0, ctx.size(0) - 1)
            m_dst = self.context_att(ctx.index_select(0, col_c))  # [E, H]
            alpha = alpha * m_dst
        if col.numel() > 0:
            alpha = torch.exp(alpha - alpha.max())
            alpha_sum = scatter_add(alpha, col, dim=0, dim_size=num_dst_nodes)
            alpha = alpha / (alpha_sum[col] + 1e-16)
        return alpha

    def propagate(self, edge_index, x, alpha, size):
        row, col = edge_index[0], edge_index[1]
        if edge_index.numel() == 0:
            return torch.zeros(size[1], x.size(1), x.size(2), device=x.device)
        out = x * alpha.unsqueeze(-1)  # [E, H, C]
        out = scatter_add(out, col, dim=0, dim_size=size[1])
        return out

class DirectedGATConv(nn.Module):
    def __init__(self, in_channels_src, in_channels_dst, out_channels, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.lin_src = nn.Linear(in_channels_src, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels_dst, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.do = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x_src, x_dst, edge_index):
        in_src = x_src.size(-1)
        in_dst = x_dst.size(-1)
        if (self.lin_src is None) or (self.lin_src.in_features != in_src):
            self.lin_src = nn.Linear(in_src, self.heads * self.out_channels, bias=False).to(x_src.device)
            nn.init.xavier_uniform_(self.lin_src.weight)
        if (self.lin_dst is None) or (self.lin_dst.in_features != in_dst):
            self.lin_dst = nn.Linear(in_dst, self.heads * self.out_channels, bias=False).to(x_dst.device)
            nn.init.xavier_uniform_(self.lin_dst.weight)
        x_src = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)
        row, col = edge_index
        alpha_src = (x_src[row] * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst[col] * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)
        alpha = softmax(alpha, row, num_nodes=x_src.size(0))
        out = scatter(x_src[row] * alpha.unsqueeze(-1), 
                      col, dim=0, dim_size=x_dst.size(0))
        return out.view(-1, self.heads * self.out_channels)

class HierarchicalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, context_dim=256, num_layers=3,
                 heads=4, attn_dropout=0.2, residual_mode: str = "outer",
                 ablation: str = "none"):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.residual_mode = residual_mode  # 'outer' | 'inner_only' | 'inner+outer' | 'off'
        self.ablation = ablation
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.expected_dim = heads * hidden_channels
        self.convs = nn.ModuleList()
        self.layer_in_dims = []  # list[dict[node_type -> in_dim]]
        node_dims = {
            'TCM_ID': in_channels, 'Protein': in_channels, 'TF': in_channels,
            'RBP': in_channels, 'TCM_symptom_ID': in_channels, 'UMLS_id': in_channels,
            'mRNA': in_channels, 'DNA': in_channels, 'lncRNA': in_channels, 'miRNA': in_channels
        }
        for _ in range(num_layers):
            self.layer_in_dims.append(dict(node_dims))  # 快照
            self.convs.append(
                self._create_conv_layer(
                    in_dims=node_dims, out_channels=hidden_channels, context_dim=context_dim
                )
            )
            for nt in node_dims:
                node_dims[nt] = self.expected_dim
        use_outer = (self.residual_mode in ("outer", "inner+outer"))
        if use_outer:
            self.res_proj = nn.ModuleList()
            for in_dims in self.layer_in_dims:
                md = nn.ModuleDict()
                for nt, d_in in in_dims.items():
                    lin = nn.Linear(d_in, self.expected_dim, bias=True)
                    nn.init.xavier_uniform_(lin.weight)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)
                    md[nt] = lin
                self.res_proj.append(md)
            all_ntypes = set()
            for snap in self.layer_in_dims:
                all_ntypes.update(snap.keys())
            self.res_norm = nn.ModuleDict({
                nt: nn.LayerNorm(self.expected_dim) for nt in sorted(all_ntypes)
            })
            self.res_drop = nn.Dropout(self.attn_dropout)
        else:
            self.res_proj = None
            self.res_norm = None
            self.res_drop = nn.Dropout(self.attn_dropout)  

    def _make_conv_for_edge(self, edge_type):
        src_type, rel, dst_type = edge_type
        use_inner = (self.residual_mode in ("inner_only", "inner+outer"))
        if (src_type == 'TCM_ID' or dst_type == 'TCM_ID' or rel.startswith('herb_')
            or src_type in ('Protein','TF','RBP','mRNA') or dst_type in ('Protein','TF','RBP','mRNA')):
            return ContextAwareGATConv(
                out_channels=self.hidden_channels,
                context_dim=self.context_dim,
                heads=self.heads,
                dropout=self.attn_dropout,
                use_inner_residual=use_inner
            )
        else:
            return GATConv(
                (-1, -1),
                out_channels=self.hidden_channels,
                heads=self.heads,
                dropout=self.attn_dropout,
                add_self_loops=False
            )

    def _create_conv_layer(self, in_dims, out_channels, context_dim,
                           local=False, global_scope=False):
        conv_dict = {}
        use_inner = (self.residual_mode in ("inner_only", "inner+outer"))
        def _ctx_conv():
            return ContextAwareGATConv(out_channels, context_dim,
                                       heads=self.heads, dropout=self.attn_dropout,
                                       use_inner_residual=use_inner)
        def _gat():
            return GATConv((-1, -1), out_channels, heads=self.heads,
                           dropout=self.attn_dropout, add_self_loops=False)
        bidirectional_relations = [
            ('TCM_ID', 'herb_treats_tcm_symptom', 'TCM_symptom_ID'),
        ]
        for rel in bidirectional_relations:
            conv_dict[rel] = _ctx_conv()
            rev = (rel[2], f"rev_{rel[1]}", rel[0])
            conv_dict[rev] = _ctx_conv()
        herb_relations = [
            ('TCM_ID', 'herb_modulates_target', 'Protein'),
            ('TCM_ID', 'herb_modulates_target', 'TF'),
            ('TCM_ID', 'herb_modulates_target', 'RBP'),
            ('TCM_ID', 'herb_upregulates_mRNA', 'mRNA'),
            ('TCM_ID', 'herb_downregulates_mRNA', 'mRNA'),
        ]
        for rel in herb_relations:
            conv_dict[rel] = _ctx_conv()
            rev = (rel[2], f"rev_{rel[1]}", rel[0])
            conv_dict[rev] = _ctx_conv()
        bio_relations = [
            ('TCM_symptom_ID', 'tcm_symptom_corresponds_to_mm_symptom', 'UMLS_id'),
            ('UMLS_id', 'mmsymptom_associated_with_gene', 'mRNA'),
        ]
        for rel in bio_relations:
            conv_dict[rel] = _gat()
            rev = (rel[2], f"rev_{rel[1]}", rel[0])
            conv_dict[rev] = _gat()
        regulatory_relations = [
            ('lncRNA', 'lncRNA_interacts_with', 'miRNA'),
            ('miRNA', 'lncRNA_interacts_with', 'lncRNA'),
            ('lncRNA', 'lncRNA_regulates', 'DNA'),
            ('miRNA', 'miRNA_regulates_mRNA', 'mRNA'),
            ('Protein', 'protein_interacts_with', 'TF'),
            ('Protein', 'protein_interacts_with', 'Protein'),
            ('TF', 'protein_interacts_with', 'Protein'),
            ('Protein', 'protein_interacts_with', 'RBP'),
            ('RBP', 'protein_interacts_with', 'Protein'),
            ('RBP', 'protein_interacts_with', 'RBP'),
            ('TF', 'protein_interacts_with', 'TF'),
            ('TF', 'protein_interacts_with', 'RBP'),
            ('RBP', 'protein_interacts_with', 'TF'),
            ('RBP', 'rbp_regulates_mRNA', 'mRNA'),
            ('TF', 'tf_regulates', 'DNA'),
            ('DNA', 'dna_transcribes_to_mRNA', 'mRNA'),
            ('mRNA', 'mRNA_translates_to_protein', 'Protein'),
            ('mRNA', 'mRNA_translates_to_protein', 'TF'),
            ('mRNA', 'mRNA_translates_to_protein', 'RBP'),
        ]
        for rel in regulatory_relations:
            src_type, _, dst_type = rel
            conv_dict[rel] = DirectedGATConv(
                in_channels_src=in_dims[src_type],
                in_channels_dst=in_dims[dst_type],
                out_channels=out_channels,
                heads=self.heads,
                dropout=self.attn_dropout,
            )
        return HeteroConv(conv_dict, aggr='mean')

    def forward(self, x_dict, edge_index_dict, node_counts,
                context_src_dict=None, context_dst_dict=None):
        for layer_idx, conv_layer in enumerate(self.convs):
            x_in = {k: v for k, v in x_dict.items()}
            out_dict = {}
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                x_src_edge = x_in.get(src_type, None)
                x_dst_edge = x_in.get(dst_type, None)
                if x_src_edge is None or x_dst_edge is None:
                    continue
                if edge_index is None or edge_index.numel() == 0:
                    continue
                row, col = edge_index[0], edge_index[1]
                need_src = max(int(row.max().item()) + 1, x_src_edge.size(0))
                need_dst = max(int(col.max().item()) + 1, x_dst_edge.size(0))
                if x_src_edge.size(0) < need_src:
                    pad = x_src_edge.new_zeros((need_src - x_src_edge.size(0), x_src_edge.size(1)))
                    x_src_edge = torch.cat([x_src_edge, pad], dim=0)
                if x_dst_edge.size(0) < need_dst:
                    pad = x_dst_edge.new_zeros((need_dst - x_dst_edge.size(0), x_dst_edge.size(1)))
                    x_dst_edge = torch.cat([x_dst_edge, pad], dim=0)
                if edge_type not in conv_layer.convs:
                    conv_layer.convs[edge_type] = self._make_conv_for_edge(edge_type).to(x_src_edge.device)
                conv = conv_layer.convs[edge_type]
                if isinstance(conv, ContextAwareGATConv):
                    ctx_src = context_src_dict.get(edge_type, None) if context_src_dict else None
                    ctx_dst = context_dst_dict.get(edge_type, None) if context_dst_dict else None
                    if ctx_src is not None and ctx_src.size(0) < need_src:
                        ctx_src = torch.cat([ctx_src,
                                             ctx_src.new_zeros((need_src - ctx_src.size(0), ctx_src.size(1)))], dim=0)
                    if ctx_dst is not None and ctx_dst.size(0) < need_dst:
                        ctx_dst = torch.cat([ctx_dst,
                                             ctx_dst.new_zeros((need_dst - ctx_dst.size(0), ctx_dst.size(1)))], dim=0)
                    out = conv(x=(x_src_edge, x_dst_edge), edge_index=edge_index,
                               context_src=ctx_src, context_dst=ctx_dst)
                elif isinstance(conv, DirectedGATConv):
                    out = conv(x_src_edge, x_dst_edge, edge_index)
                else:
                    out = conv(x=(x_src_edge, x_dst_edge), edge_index=edge_index) \
                          if src_type != dst_type else conv(x=x_src_edge, edge_index=edge_index)
                out_dict.setdefault(dst_type, []).append(out)
            for nt in out_dict:
                out_dict[nt] = torch.stack(out_dict[nt], dim=0).mean(dim=0)
            if self.ablation == "res_only":
                for nt in out_dict:
                    out_dict[nt] = torch.zeros_like(out_dict[nt])
            use_outer = (self.res_proj is not None)
            updated = {}
            all_nt = set(list(x_in.keys()) + list(out_dict.keys()))
            for nt in all_nt:
                y = out_dict.get(nt, None)
                if y is None:
                    y = x_in[nt].new_zeros((x_in[nt].size(0), self.expected_dim))
                if use_outer:
                    res_term = self.res_proj[layer_idx][nt](x_in[nt])  # [*, expected_dim]
                    y = y + res_term
                    y = self.res_norm[nt](self.res_drop(y))
                updated[nt] = y
            for nt in list(updated.keys()):
                expected_num = node_counts.get(nt, updated[nt].size(0))
                n, d = updated[nt].size(0), updated[nt].size(1)
                if n < expected_num:
                    pad = updated[nt].new_zeros((expected_num - n, d))
                    updated[nt] = torch.cat([updated[nt], pad], dim=0)
                elif n > expected_num:
                    updated[nt] = updated[nt][:expected_num]
            for nt, mat in updated.items():
                assert mat.size(-1) == self.expected_dim, \
                    f"{nt} output dim={mat.size(-1)} != expected_dim={self.expected_dim}"
            x_dict = updated
        return x_dict
    
class FusionEncoder(nn.Module):
    def __init__(self, text_dim, nes_dim, context_dim=256, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + nes_dim, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden),
            nn.Linear(hidden, context_dim),
            nn.GELU(), nn.LayerNorm(context_dim),
        )
    def forward(self, S, N):
        return self.net(torch.cat([S, N], dim=-1))

class MAGED(nn.Module):
    def __init__(self, node_counts, hidden_channels=256, num_gnn_layers=3, 
                 pretrained_embeddings=None, context_dim=256,
                 num_heads=4, attn_dropout=0.2, in_dim=None,
                 fusion_encoder: nn.Module = None,
                 scorer_name: str = "Concat_MLP",
                 scorer_kwargs: dict = None,
                 residual_mode: str = "inner+outer", 
                ablation: str ="none"            
                ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.in_dim = int(in_dim) if in_dim is not None else int(hidden_channels)
        self.fusion_encoder = fusion_encoder
        self.scorer_name = scorer_name
        self.scorer_kwargs = scorer_kwargs or {}
        self.target_predictor = None  
        self.embedding = nn.ModuleDict()
        self.idmap = {}
        self.pre_proj = nn.ModuleDict()
        self.tcm_fusion = nn.Linear(self.in_dim + context_dim, hidden_channels)
        self.pre_gnn_proj = (nn.Identity() if self.in_dim == self.hidden_channels
                     else nn.Linear(self.in_dim, self.hidden_channels))
        for node_type, count in node_counts.items():
            if pretrained_embeddings is not None and node_type in pretrained_embeddings:
                emb_tensor, id_mapping = pretrained_embeddings[node_type]
                print(f"[INFO] {node_type} 预训练 embedding 加载成功, shape={emb_tensor.shape}")
                emb_layer = nn.Embedding.from_pretrained(emb_tensor, freeze=False)
                self.embedding[node_type] = emb_layer
                self.idmap[node_type] = id_mapping
                pre_dim = emb_tensor.size(1)
                if pre_dim != self.in_dim:
                    self.pre_proj[node_type] = nn.Linear(pre_dim, self.in_dim)
            else:
                self.embedding[node_type] = nn.Embedding(count, self.in_dim)
                self.idmap[node_type] = {}
        self.gnn = HierarchicalGNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            context_dim=context_dim,
            num_layers=num_gnn_layers,
            heads=num_heads,
            attn_dropout=attn_dropout,
            residual_mode=residual_mode, 
            ablation=ablation,
            )

    def _lookup_embed(self, node_type, entity_names, data, device):
        num_nodes = data[node_type].num_nodes
        if num_nodes == 0:
            return torch.zeros((0, self.in_dim), device=device)
        id_mapping = self.idmap[node_type]
        emb_layer = self.embedding[node_type]
        need_fill = (not id_mapping) or any(n not in id_mapping for n in entity_names)
        if need_fill:
            eid = data[node_type].entity_id
            eid_list = eid.tolist() if isinstance(eid, torch.Tensor) else list(eid)
            max_eid = max(int(i) for i in eid_list)
            id_mapping.update({str(n): int(i) for n, i in zip(entity_names, eid_list)})
        indices = torch.tensor([id_mapping[e_name] for e_name in entity_names], dtype=torch.long, device=device)
        emb = emb_layer(indices)
        if node_type in self.pre_proj:
            emb = self.pre_proj[node_type](emb)
        return emb

    def _get_context(self, data, device):
        S = data['TCM_ID'].text_feat.to(device)
        N = data['TCM_ID'].nes_feat.to(device)
        if self.fusion_encoder is None:
            return torch.zeros(S.size(0), self.context_dim, device=device)
        return self.fusion_encoder(S, N)

    def forward(self, data, entity_name_dict):
        device = next(self.parameters()).device
        x_dict = {}
        for node_type in data.node_types:
            entity_names = entity_name_dict[node_type]
            x_dict[node_type] = self._lookup_embed(node_type, entity_names, data, device)
        c_h = self._get_context(data, device)  # [H, context_dim]
        if 'TCM_ID' in x_dict and x_dict['TCM_ID'].size(0) > 0:
            tcm = x_dict['TCM_ID']
            n = min(tcm.size(0), c_h.size(0))
            fused = self.tcm_fusion(torch.cat([tcm[:n], c_h[:n]], dim=-1))
            if n < tcm.size(0):
                pad = torch.zeros(tcm.size(0)-n, fused.size(1), device=device, dtype=fused.dtype)
                fused = torch.cat([fused, pad], dim=0)
            x_dict['TCM_ID'] = fused
        context_src_dict, context_dst_dict = {}, {}
        for et in data.edge_index_dict.keys():
            src_t, _, dst_t = et
            if src_t == 'TCM_ID':
                context_src_dict[et] = c_h
            if dst_t == 'TCM_ID':
                context_dst_dict[et] = c_h
        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_nodes = data[src_type].num_nodes
            dst_nodes = data[dst_type].num_nodes
            edge_index[0] = torch.clamp(edge_index[0], 0, src_nodes - 1)
            edge_index[1] = torch.clamp(edge_index[1], 0, dst_nodes - 1)
            data.edge_index_dict[edge_type] = edge_index
        for nt, x in x_dict.items():
            if nt == 'TCM_ID':            
                continue
            x_dict[nt] = self.pre_gnn_proj(x)
        embeddings = self.gnn(
            x_dict,
            data.edge_index_dict,
            {nt: data[nt].num_nodes for nt in data.node_types},  
            context_src_dict,
            context_dst_dict,
        )
        for nt in ['TCM_ID', 'Protein', 'TF', 'RBP', 'TCM_symptom_ID']:
            if nt not in embeddings and nt in x_dict:
                embeddings[nt] = x_dict[nt]
        herb_emb = embeddings['TCM_ID']
        target_parts = [embeddings[t] for t in TARGET_TYPES if t in embeddings]
        target_emb = target_parts[0] if len(target_parts) == 1 else torch.cat(target_parts, dim=0)
        if self.target_predictor is None:
            d = herb_emb.size(-1)
            self.target_predictor = build_scorer(self.scorer_name, dim=d, **self.scorer_kwargs).to(herb_emb.device)
        return self.predict_links(herb_emb, target_emb, self.target_predictor)

    def predict_links(self, src_emb, dst_emb, predictor, chunk_src: int = 1024, chunk_dst: int = 32768):
        device = src_emb.device
        B, N = src_emb.size(0), dst_emb.size(0)
        if B == 0 or N == 0:
            return torch.zeros((B, N), device=device, dtype=src_emb.dtype)
        out = []
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
            for i in range(0, B, chunk_src):
                s = src_emb[i:i+chunk_src]           # [b,d]
                row = []
                for j in range(0, N, chunk_dst):
                    t = dst_emb[j:j+chunk_dst]       # [n,d]
                    if hasattr(predictor, "score_matrix"):
                        scores = predictor.score_matrix(s, t)           # [b,n]
                    else:
                        scores = s @ t.T
                    row.append(scores.to(dtype=src_emb.dtype))
                out.append(torch.cat(row, dim=1))
        return torch.cat(out, dim=0)

LOSS_FN_REGISTRY = {
    'BPR': bpr_pairwise_loss,
    'Lambdarank': lambdarank_ndcg_approx_loss,
    'BCE': bce_loss,
    'Margin': margin_pairwise_loss,
    'Softmax': softmax_loss,
}
def compute_target_loss(  
        pred,
        batch,
        mask_dict=None,
        pos_set=None,
        num_neg_per_pos=5,
        loss_type='BPR',
        sampling_strategy='uniform',      # 'uniform' | 'popularity' | 'semi-hard' | 'bern_local' | 'self_adversarial'
        popularity_weight=None,  
        semi_hard_pool=100,
        adv_tau=2.0,
        global_pos_dict=None,
    ):
    device = pred.device
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0
    target_types = [t for t in TARGET_TYPES if t in batch.node_types]
    et_rel = 'herb_modulates_target'
    src = 'TCM_ID'
    sizes, offsets, ptr = {}, {}, 0
    for t in target_types:
        n = batch[t].num_nodes if t in batch.node_types else 0
        sizes[t] = n
        offsets[t] = ptr
        ptr += n
    sum_ptr = ptr
    use_concat_cols = (pred.size(1) == sum_ptr and sum_ptr > 0)
    gidx_src = getattr(batch['TCM_ID'], 'global_indices', None)
    gidx_tgt = {t: getattr(batch[t], 'global_indices', None) if sizes[t] > 0 else None
                for t in target_types}
    def resolve_pop_local(ttype: str):
        if popularity_weight is None:
            return None
        w = popularity_weight.get(ttype)
        if w is None:
            return None
        w = w.to(device)
        if w.numel() == sizes[ttype]:
            return w
        g = getattr(batch[ttype], 'global_indices', None)
        if g is not None and w.numel() >= int(g.max().item()) + 1:
            return w[g.to(device)]
        return None
    for tgt in target_types:
        n_type = sizes[tgt]
        if n_type == 0:
            continue
        et = (src, et_rel, tgt)
        if et not in batch.edge_index_dict:
            continue
        storage = batch[et]
        edge_index = storage.edge_index
        if mask_dict is not None and et in mask_dict and mask_dict[et] is not None:
            edge_index = edge_index[:, mask_dict[et]]
        if edge_index.numel() == 0:
            continue
        pop_w_local = resolve_pop_local(tgt) if sampling_strategy == 'popularity' else None
        if sampling_strategy == 'bern_local':
            deg = torch.zeros(n_type, dtype=torch.float32, device=device)
            t_local_all = edge_index[1].to(device)
            deg.index_add_(0, t_local_all, torch.ones_like(t_local_all, dtype=torch.float32, device=device))
            maxd = torch.clamp(deg.max(), min=0.0)
            w = (maxd + 1.0) - deg
            if torch.all(w <= 0):
                bern_prob_local = torch.full((n_type,), 1.0 / max(1, n_type), device=device)
            else:
                bern_prob_local = w / (w.sum() + 1e-12)
        else:
            bern_prob_local = None
        all_local = torch.arange(n_type, device=device)
        col_offset = offsets[tgt] if use_concat_cols else 0
        gid_to_local = None
        if gidx_tgt[tgt] is not None:
            gids = gidx_tgt[tgt].tolist()
            gid_to_local = {int(g): i for i, g in enumerate(gids)}
        h_local = edge_index[0].to(device)
        t_local = edge_index[1].to(device)
        order = torch.argsort(h_local)                       # O(E log E)，但 E 为该 batch 的边数
        h_sorted = h_local[order]
        t_sorted = t_local[order]
        uniq_h, counts = torch.unique_consecutive(h_sorted, return_counts=True)
        start = torch.zeros_like(counts)
        start[1:] = torch.cumsum(counts, dim=0)[:-1]
        for i in range(uniq_h.numel()):
            h = int(uniq_h[i].item())
            st = int(start[i].item())
            ed = st + int(counts[i].item())
            pos_local = t_sorted[st:ed]                      # 1D, L_pos > 0
            if pos_local.numel() == 0:
                continue
            need_k = max(1, int(num_neg_per_pos) * int(pos_local.numel()))
            with torch.no_grad():
                if global_pos_dict is not None and gidx_src is not None and gidx_tgt[tgt] is not None:
                    gh = int(gidx_src[h].item())
                    g_pos_set = global_pos_dict.get(tgt, {}).get(gh, None)
                else:
                    g_pos_set = None
                if g_pos_set and gid_to_local is not None:
                    extra = [gid_to_local[g] for g in g_pos_set if g in gid_to_local]
                    if len(extra) > 0:
                        exclude_local = torch.unique(torch.cat([pos_local, torch.tensor(extra, device=device, dtype=torch.long)]))
                    else:
                        exclude_local = pos_local
                else:
                    exclude_local = pos_local
                neg_mask = torch.ones(n_type, dtype=torch.bool, device=device)
                neg_mask[exclude_local] = False
                neg_pool = all_local[neg_mask]               # 1D
                if neg_pool.numel() == 0:
                    continue
                if sampling_strategy == 'popularity' and pop_w_local is not None:
                    w = pop_w_local[neg_pool]
                    if (not torch.isfinite(w).all()) or torch.all(w <= 0):
                        w = torch.ones_like(w)
                    prob = w / (w.sum() + 1e-12)
                    choose_k = min(need_k, neg_pool.numel())
                    picked_local = neg_pool[torch.multinomial(prob, num_samples=choose_k, replacement=False)]
                elif sampling_strategy == 'semi-hard':
                    pool_k = min(int(semi_hard_pool), neg_pool.numel())
                    if pool_k <= 0:
                        continue
                    cand_local = neg_pool[torch.randperm(neg_pool.numel(), device=device)[:pool_k]]
                    cand_cols = cand_local + col_offset
                    cand_scores = pred[h, cand_cols].detach()
                    topk = min(need_k, cand_scores.numel())
                    if topk <= 0:
                        continue
                    _, idx = torch.topk(cand_scores, k=topk, largest=True)
                    picked_local = cand_local[idx]
                elif sampling_strategy == 'self_adversarial':
                    pool_k = min(int(semi_hard_pool), neg_pool.numel())
                    if pool_k <= 0:
                        continue
                    cand_local = neg_pool[torch.randperm(neg_pool.numel(), device=device)[:pool_k]]
                    cand_cols = cand_local + col_offset
                    w = torch.exp(adv_tau * pred[h, cand_cols].detach())
                    if (not torch.isfinite(w).all()) or torch.all(w <= 0):
                        w = torch.ones_like(w)
                    prob = w / (w.sum() + 1e-12)
                    choose_k = min(need_k, cand_local.numel())
                    picked_local = cand_local[torch.multinomial(prob, num_samples=choose_k, replacement=False)]
                elif sampling_strategy == 'bern_local':
                    w = bern_prob_local[neg_pool] if bern_prob_local is not None else None
                    if (w is None) or (not torch.isfinite(w).all()) or torch.all(w <= 0):
                        w = torch.ones(neg_pool.numel(), device=device)
                    prob = w / (w.sum() + 1e-12)
                    choose_k = min(need_k, neg_pool.numel())
                    picked_local = neg_pool[torch.multinomial(prob, num_samples=choose_k, replacement=False)]
                else:
                    choose_k = min(need_k, neg_pool.numel())
                    picked_local = neg_pool[torch.randperm(neg_pool.numel(), device=device)[:choose_k]]
            if picked_local.numel() == 0:
                continue
            pos_cols  = pos_local + col_offset
            neg_cols  = picked_local + col_offset
            pos_scores = pred[h, pos_cols]
            neg_scores = pred[h, neg_cols]
            total_loss = total_loss + LOSS_FN_REGISTRY[loss_type](pos_scores, neg_scores)
            valid_count += 1
    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / valid_count

def calculate_hr_per_herb(herb_scores, true_labels, k=10):
    herb_scores = herb_scores.to(true_labels.device)
    _, indices = torch.sort(herb_scores, descending=True)
    topk_indices = indices[:k]
    hit = (true_labels[topk_indices].sum() > 0).float().item()
    return hit

def calculate_recall_per_herb(herb_scores, true_labels, k=100):
    herb_scores = herb_scores.to(true_labels.device)
    _, indices = torch.sort(herb_scores, descending=True)
    valid_k = min(k, len(indices))
    topk_indices = indices[:valid_k]
    max_index = len(true_labels) - 1
    topk_indices = torch.clamp(topk_indices, 0, max_index)
    relevant = true_labels[topk_indices].sum().item()
    total_positive = true_labels.sum().item()
    recall = relevant / total_positive if total_positive > 0 else 0.0
    return recall

def calculate_ndcg_per_herb(herb_scores: torch.Tensor,
                            true_labels: torch.Tensor,
                            k: int = 10) -> float:
    herb_scores = herb_scores.to(true_labels.device)
    n = herb_scores.numel()
    if n == 0:
        return 0.0
    k = min(k, n)
    topk_idx = torch.argsort(herb_scores, descending=True)[:k]
    rel_topk = true_labels[topk_idx]                       # [k], 0/1
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=true_labels.device))
    dcg = (rel_topk.float() / discounts).sum().item()
    P = int(true_labels.sum().item())
    if P == 0:
        return 0.0
    ideal_k = min(k, P)
    ideal_rel= torch.cat([
        torch.ones(ideal_k,dtype=torch.float32,device=true_labels.device),
        torch.zeros(max(0,k-ideal_k),dtype=torch.float32,device=true_labels.device),       
    ])
    idcg = (ideal_rel / discounts[:len(ideal_rel)]).sum().item()
    return dcg / idcg

def evaluate_model(model, data, full_entity_id_map, mask_type='test'):
    try:
        data_device = data['TCM_ID'].x.device
    except Exception:
        data_device = next(model.parameters()).device

    if next(model.parameters()).device != data_device:
        model.to(data_device)
    model.eval()
    with torch.no_grad():
        entity_name_dict = batch_entity_names(data, full_entity_id_map)
        target_pred = model(data, entity_name_dict)
        size_map = {t: (data[t].num_nodes if t in data.node_types else 0) for t in TARGET_TYPES}
        all_types = [t for t in TARGET_TYPES if size_map.get(t, 0) > 0]
        pred_cols = target_pred.size(1)
        single = [t for t in all_types if size_map[t] == pred_cols]
        if len(single) == 1:
            target_kind = single[0]     # 例如只评 Protein 或只评 TF
        else:
            target_kind = 'Target'      # 合并评估（多类型拼接）
        target_metrics = compute_metrics_by_herb(
            target_pred, data,
            'herb_modulates_target', target_kind,
            mask_type=mask_type
        )
        for target_type in TARGET_TYPES:
            edge_type = ('TCM_ID', 'herb_modulates_target', target_type)
            if edge_type in data.edge_types:
                if mask_type == 'train':
                    mask = data[edge_type].train_mask
                elif mask_type == 'val':
                    mask = data[edge_type].val_mask
                elif mask_type == 'test':
                    mask = data[edge_type].test_mask
                else:
                    mask = None
                    
                if mask is not None:
                    count = mask.sum().item()
                else:
                    print(f"{edge_type}: 无{mask_type}掩码")
        return target_metrics

def compute_metrics_by_herb(pred, data, relation, target_type, mask_type='test'):
    herb_indices = data['TCM_ID'].node_id
    num_herbs = len(herb_indices)
    print(f"\n开始评估 {target_type} 预测 ({mask_type}集) ...")
    print(f"中药节点数量: {num_herbs}")
    hr1_sum = hr3_sum = hr5_sum = hr10_sum = hr50_sum = 0.0
    ndcg1_sum = ndcg3_sum = ndcg5_sum = ndcg10_sum = ndcg50_sum = 0.0
    recall1_sum = recall10_sum = recall50_sum = recall100_sum = recall200_sum = recall300_sum = recall500_sum = 0.0
    valid_herbs = 0
    size_map = {t: (data[t].num_nodes if t in data.node_types else 0) for t in TARGET_TYPES}
    all_types = [t for t in TARGET_TYPES if size_map.get(t, 0) > 0]
    pred_cols = pred.size(1)
    if target_type == 'Target':
        total_full = sum(size_map[t] for t in all_types)
        if total_full > 0 and pred_cols == total_full:
            used_types = all_types
        else:
            single = [t for t in all_types if size_map[t] == pred_cols]
            if len(single) == 1:
                used_types = single
            else:
                used_types, acc = [], 0
                for t in all_types:
                    acc += size_map[t]
                    used_types.append(t)
                    if acc == pred_cols:
                        break
                if sum(size_map[t] for t in used_types) != pred_cols:
                    used_types = all_types
        offsets, ptr = {}, 0
        for t in used_types:
            offsets[t] = ptr
            ptr += size_map[t]
        total_targets = ptr
    else:
        used_types = [target_type]
        offsets = {target_type: 0}
        total_targets = pred_cols
    def get_masks(et):
        m_train = getattr(data[et], 'train_mask', None)
        m_val   = getattr(data[et], 'val_mask',   None)
        m_test  = getattr(data[et], 'test_mask',  None)
        m_eval = None
        if mask_type == 'train': m_eval = m_train
        elif mask_type == 'val': m_eval = m_val
        elif mask_type == 'test': m_eval = m_test
        masks = [m for m in (m_train, m_val, m_test) if m is not None]
        m_all = None
        if len(masks) > 0:
            m_all = masks[0].clone()
            for m in masks[1:]:
                m_all |= m
        return m_eval, m_all
    for herb_idx in herb_indices:
        herb_scores = pred[herb_idx].clone() 
        device = herb_scores.device
        true_labels = torch.zeros(total_targets, device=device)
        has_positive = False
        herb_idx_int = int(herb_idx.item() if isinstance(herb_idx, torch.Tensor) else herb_idx)
        eval_pos_cols_all = [] 
        other_pos_cols_all = []
        for t in used_types:
            if t not in data.node_types:
                continue
            et = ('TCM_ID', relation, t)
            off = offsets[t]
            if et not in data.edge_types:
                continue
            edge_index = data[et].edge_index
            m_eval, m_all = get_masks(et)
            if m_eval is not None:
                ei_eval = edge_index[:, m_eval]
            else:
                ei_eval = edge_index[:, []]
            eval_targets_local = ei_eval[1][ei_eval[0] == herb_idx_int]  # 该 herb 的（本划分）目标局部索引
            if m_all is not None:
                ei_all = edge_index[:, m_all]
            else:
                ei_all = edge_index
            all_targets_local = ei_all[1][ei_all[0] == herb_idx_int]
            if eval_targets_local.numel() > 0:
                others = set(all_targets_local.tolist()) - set(eval_targets_local.tolist())
            else:
                others = set(all_targets_local.tolist())
            off = offsets[t]
            if eval_targets_local.numel() > 0:
                eval_pos_cols_all.append(eval_targets_local.to(device) + off)
            if len(others) > 0:
                other_pos_cols_all.append(torch.tensor(list(others), device=device, dtype=torch.long) + off)
        if len(eval_pos_cols_all) > 0:
            eval_pos_cols = torch.cat(eval_pos_cols_all, dim=0)
            true_labels[eval_pos_cols] = 1.0
            has_positive = True
        if not has_positive:
            continue
        if len(other_pos_cols_all) > 0:
            other_pos_cols = torch.unique(torch.cat(other_pos_cols_all, dim=0))
            herb_scores[other_pos_cols] = float('-inf')
        hr1   = calculate_hr_per_herb(herb_scores, true_labels, k=1)
        hr3   = calculate_hr_per_herb(herb_scores, true_labels, k=3)
        hr5   = calculate_hr_per_herb(herb_scores, true_labels, k=5)
        hr10  = calculate_hr_per_herb(herb_scores, true_labels, k=10)
        hr50  = calculate_hr_per_herb(herb_scores, true_labels, k=50)
        ndcg1  = calculate_ndcg_per_herb(herb_scores, true_labels, k=1)
        ndcg3  = calculate_ndcg_per_herb(herb_scores, true_labels, k=3)
        ndcg5  = calculate_ndcg_per_herb(herb_scores, true_labels, k=5)
        ndcg10 = calculate_ndcg_per_herb(herb_scores, true_labels, k=10)
        ndcg50 = calculate_ndcg_per_herb(herb_scores, true_labels, k=50)
        recall1  = calculate_recall_per_herb(herb_scores, true_labels, k=1)
        recall10  = calculate_recall_per_herb(herb_scores, true_labels, k=10)
        recall50  = calculate_recall_per_herb(herb_scores, true_labels, k=50)
        recall100 = calculate_recall_per_herb(herb_scores, true_labels, k=100)
        recall200 = calculate_recall_per_herb(herb_scores, true_labels, k=200)
        recall300 = calculate_recall_per_herb(herb_scores, true_labels, k=300)
        recall500 = calculate_recall_per_herb(herb_scores, true_labels, k=500)
        hr1_sum += hr1; hr3_sum += hr3; hr5_sum += hr5; hr10_sum += hr10; hr50_sum += hr50
        ndcg1_sum += ndcg1; ndcg3_sum += ndcg3; ndcg5_sum += ndcg5; ndcg10_sum += ndcg10; ndcg50_sum += ndcg50
        recall1_sum += recall1; recall10_sum += recall10; recall50_sum += recall50; recall100_sum += recall100
        recall200_sum += recall200; recall300_sum += recall300; recall500_sum += recall500
        valid_herbs += 1
    if valid_herbs > 0:
        return {
            'hr@1': hr1_sum / valid_herbs,
            'hr@3': hr3_sum / valid_herbs,
            'hr@5': hr5_sum / valid_herbs,
            'hr@10': hr10_sum / valid_herbs,
            'hr@50': hr50_sum / valid_herbs,
            'ndcg@1': ndcg1_sum / valid_herbs,
            'ndcg@3': ndcg3_sum / valid_herbs,
            'ndcg@5': ndcg5_sum / valid_herbs,
            'ndcg@10': ndcg10_sum / valid_herbs,
            'ndcg@50': ndcg50_sum / valid_herbs,
            'recall@1': recall1_sum / valid_herbs,
            'recall@10': recall10_sum / valid_herbs,
            'recall@50': recall50_sum / valid_herbs,
            'recall@100': recall100_sum / valid_herbs,
            'recall@200': recall200_sum / valid_herbs,
            'recall@300': recall300_sum / valid_herbs,
            'recall@500': recall500_sum / valid_herbs,
            'num_herbs': valid_herbs
        }
    else:
        return {
            'hr@10': 0.0, 'hr@50': 0.0, 'ndcg@10': 0.0, 'ndcg@50': 0.0,
            'recall@100': 0.0, 'recall@500': 0.0, 'num_herbs': 0
        }


def save_loss_plot(train_losses, val_losses, out_png, title="Loss Curve"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()
