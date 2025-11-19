import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split, KFold
from collections import defaultdict
from .encoder import *
from .utils_embedding import TARGET_TYPES
from .maged import *

def prune_to_largest_cc(data):
    full_graph = data.to_homogeneous(node_attrs=['global_indices'])
    G = nx.Graph()
    G.add_nodes_from(range(full_graph.num_nodes))
    G.add_edges_from(full_graph.edge_index.t().tolist())
    largest_cc = max(nx.connected_components(G), key=len)
    cc_indices = list(largest_cc)
    global_indices_all = full_graph.global_indices.cpu().numpy()
    cc_global_indices = set(global_indices_all[i] for i in cc_indices)
    pruned_data = HeteroData()
    for node_type in data.node_types:
        node_global_indices = data[node_type].global_indices.cpu().numpy()
        mask = np.array([g in cc_global_indices for g in node_global_indices])
        if mask.sum() == 0:
            continue
        pruned_data[node_type].num_nodes = mask.sum()
        pruned_data[node_type].global_indices = torch.tensor(node_global_indices[mask], dtype=torch.long)
        pruned_data[node_type].node_id = torch.arange(mask.sum(), dtype=torch.long)
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        if src_type not in pruned_data.node_types or dst_type not in pruned_data.node_types:
            continue
        edge_index = data[edge_type].edge_index
        num_src_nodes = data[src_type].global_indices.shape[0]
        num_dst_nodes = data[dst_type].global_indices.shape[0]
        valid_mask = (edge_index[0] < num_src_nodes) & (edge_index[1] < num_dst_nodes)
        edge_index = edge_index[:, valid_mask]
        if edge_index.numel() == 0:
            continue
        src_globals = data[src_type].global_indices[edge_index[0]].cpu().numpy()
        dst_globals = data[dst_type].global_indices[edge_index[1]].cpu().numpy()
        mask = np.array([(s in cc_global_indices) and (d in cc_global_indices) for s, d in zip(src_globals, dst_globals)])
        if mask.sum() == 0:
            continue
        old2new_src = {g: i for i, g in enumerate(pruned_data[src_type].global_indices.cpu().numpy())}
        old2new_dst = {g: i for i, g in enumerate(pruned_data[dst_type].global_indices.cpu().numpy())}
        new_edges = []
        for s, d in zip(src_globals[mask], dst_globals[mask]):
            new_edges.append([old2new_src[s], old2new_dst[d]])
        pruned_data[edge_type].edge_index = torch.tensor(new_edges, dtype=torch.long).t()
    return pruned_data



class KnowledgeGraphBuilder:
    def __init__(self, data_dir, herb_data_dict):
        self.data_dir = data_dir
        self.herb_data_dict = herb_data_dict
        self.relation_files = {
            "herb_upregulates_mRNA": "herb_mRNA_triples.tsv",
            "herb_downregulates_mRNA": "herb_mRNA_triples.tsv",
            "herb_treats_tcm_symptom": "herb_symptom_triples.tsv",
            "herb_modulates_target": "herb_target_triples.tsv",
            "lncRNA_interacts_with": "LMI_triples.tsv",
            "lncRNA_regulates": "lncRNA_Regulates_triples.tsv",
            "miRNA_regulates_mRNA": "miRNA_Regulates_triples.tsv",
            "mmsymptom_associated_with_gene": "MMsymptom_gene_triples.tsv",
            "protein_interacts_with": "PPI_triples.tsv",
            "rbp_regulates_mRNA": "RBP_Regulates_triples.tsv",
            "tcm_symptom_corresponds_to_mm_symptom": "TCMsymptom_MMsymptom_triples.tsv",
            "tf_regulates": "TF_Regulates_triples.tsv",
            "dna_transcribes_to_mRNA": "Transcribes_triples.tsv",
            "mRNA_translates_to_protein": "Translates_triples.tsv"
        }
        self.triples = []
        self.entity_types = {}
        self.relation_stats = defaultdict(int)
        self.entity_id_map = {}
        self.reverse_entity_map = {}
        self.type_start_idx = {} 
        for tcm_id in herb_data_dict.keys():
            self._add_entity(f"{tcm_id}", "TCM_ID")
    
    def _add_entity(self, entity_id, entity_type):
        if entity_id not in self.entity_id_map:
            idx = len(self.entity_id_map)
            self.entity_id_map[entity_id] = idx
            self.reverse_entity_map[idx] = entity_id
            self.entity_types[entity_id] = entity_type
            if entity_type not in self.type_start_idx:
                self.type_start_idx[entity_type] = idx
        return self.entity_id_map[entity_id]
    
    def _process_herb_mrna_file(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, sep=',', header=0, names=['head', 'relation', 'tail'])
        for _, row in df.iterrows():
            head = row["head"]
            tail = row["tail"]
            relation = row["relation"]
            if "upregulates" in relation:
                rel_type = "herb_upregulates_mRNA"
            elif "downregulates" in relation:
                rel_type = "herb_downregulates_mRNA"
            else:
                continue
            head_id = self._add_entity(head, "TCM_ID")
            tail_id = self._add_entity(tail, "mRNA")
            self.triples.append((head_id, tail_id, rel_type))
            self.relation_stats[rel_type] += 1
    
    def _process_tcm_mm_symptom_file(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, sep=',', header=0, names=['head', 'relation', 'tail'])
        for _, row in df.iterrows():
            head = row["head"]
            tail = row["tail"]
            relation = row["relation"]
            rel_type = "tcm_symptom_corresponds_to_mm_symptom"
            head_id = self._add_entity(head, "TCM_symptom_ID")
            tail_id = self._add_entity(tail, "UMLS_id")
            self.triples.append((head_id, tail_id, rel_type))
            self.relation_stats[rel_type] += 1
            reverse_relation = f"rev_{rel_type}"
            self.triples.append((tail_id, head_id, reverse_relation))
            self.relation_stats[reverse_relation] += 1
    
    def _process_generic_file(self, file_path, relation_type):
        try:
            df = pd.read_csv(file_path, sep='\t', header=0, names=['head', 'relation', 'tail'])
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, sep=',', header=0, names=['head', 'relation', 'tail'])
        for _, row in df.iterrows():
            head = row["head"]
            tail = row["tail"]
            head_type = self._infer_entity_type(head, relation_type)
            tail_type = self._infer_entity_type(tail, relation_type)
            head_id = self._add_entity(head, head_type)
            tail_id = self._add_entity(tail, tail_type)
            self.triples.append((head_id, tail_id, relation_type))
            self.relation_stats[relation_type] += 1
    
    def _infer_entity_type(self, entity_id, relation_type):
        if ":" in entity_id:
            return entity_id.split(":")[0]
    
    def load_all_triples(self):
        print("Loading knowledge graph triplets...")
        herb_mrna_path = os.path.join(self.data_dir, self.relation_files["herb_upregulates_mRNA"])
        self._process_herb_mrna_file(herb_mrna_path)
        tcm_mm_path = os.path.join(self.data_dir, self.relation_files["tcm_symptom_corresponds_to_mm_symptom"])
        self._process_tcm_mm_symptom_file(tcm_mm_path)
        for rel_type, file_name in self.relation_files.items():
            if rel_type in ["herb_upregulates_mRNA", "herb_downregulates_mRNA", 
                           "tcm_symptom_corresponds_to_mm_symptom"]:
                continue
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path):
                print(f"Processing {rel_type} relation: {file_name}")
                self._process_generic_file(file_path, rel_type)
        print(f"Total loaded {len(self.triples)} triplets")
        print(f"Entity count: {len(self.entity_id_map)}")
        print(f"Relation statistics: {dict(self.relation_stats)}")
    
    def build_hetero_graph(self):
        print("\nBuilding heterogeneous graph...")
        data = HeteroData()
        node_type_to_global_indices = defaultdict(list)
        for entity_id, global_idx in self.entity_id_map.items():
            node_type = self.entity_types[entity_id]
            node_type_to_global_indices[node_type].append(global_idx)
        self.local_idx_to_entity_name = {} 
        for node_type, global_indices in node_type_to_global_indices.items():
            num_nodes = len(global_indices)
            data[node_type].num_nodes = num_nodes
            data[node_type].global_indices = torch.tensor(global_indices, dtype=torch.long)
            data[node_type].entity_id = torch.arange(num_nodes, dtype=torch.long)  
            self.local_idx_to_entity_name[node_type] = [
                self.reverse_entity_map[gidx] for gidx in global_indices
            ]
            print(f"Adding node type: {node_type}, count: {num_nodes}")
        type_to_local_index = {}
        for node_type, global_indices in node_type_to_global_indices.items():
            sorted_indices = sorted(global_indices)
            type_to_local_index[node_type] = {g_idx: i for i, g_idx in enumerate(sorted_indices)}
        
        SYMMETRIC_REL = [
            'herb_treats_tcm_symptom',
            'tcm_symptom_corresponds_to_mm_symptom',
            'mmsymptom_associated_with_gene','herb_modulates_target', 
            "herb_upregulates_mRNA", "herb_downregulates_mRNA",
        ]
        CAUSAL = {
            'tf_regulates', 'miRNA_regulates_mRNA', 'lncRNA_regulates',
            'rbp_regulates_mRNA', 'dna_transcribes_to_mRNA',
            'mRNA_translates_to_protein',
        }
        edge_index_dict = {}
        for head_idx, tail_idx, rel_type in self.triples:
            head_entity = self.reverse_entity_map[head_idx]
            tail_entity = self.reverse_entity_map[tail_idx]
            head_type = self.entity_types[head_entity]
            tail_type = self.entity_types[tail_entity]
            head_local_idx = type_to_local_index[head_type].get(head_idx)
            tail_local_idx = type_to_local_index[tail_type].get(tail_idx)
            if head_local_idx is None or tail_local_idx is None:
                print(f"Warning: Cannot find local index head_idx={head_idx} head_type={head_type} or tail_idx={tail_idx} tail_type={tail_type}")
                continue
            edge_type = (head_type, rel_type, tail_type)
            if edge_type not in edge_index_dict:
                edge_index_dict[edge_type] = [[], []]
            edge_index_dict[edge_type][0].append(head_local_idx)
            edge_index_dict[edge_type][1].append(tail_local_idx)
        for edge_type, edge_index in edge_index_dict.items():
            head_type, rel_type, tail_type = edge_type
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            data[head_type, rel_type, tail_type].edge_index = edge_index_tensor
            print(f"Adding edge: {head_type} -[{rel_type}]-> {tail_type}, count: {edge_index_tensor.size(1)}")
        for (head_type, rel_type, tail_type), edge_index in list(edge_index_dict.items()):
            if rel_type in SYMMETRIC_REL and not rel_type.startswith('rev_'):
                rev_rel = f"rev_{rel_type}"
                et_rev = (tail_type, rev_rel, head_type)
                if et_rev not in data.edge_types:
                    ei = torch.tensor(edge_index, dtype=torch.long)
                    ei_rev = ei.flip(0)  # Reverse
                    data[et_rev].edge_index = ei_rev
                    print(f"Adding reverse edge: {tail_type} -[{rev_rel}]-> {head_type}, count: {ei_rev.size(1)}")
        try:
            pruned_data = prune_to_largest_cc(data)
            for node_type in pruned_data.node_types: 
                num_nodes = pruned_data[node_type].num_nodes
                pruned_data[node_type].node_id = torch.arange(num_nodes, dtype=torch.long)
                pruned_global_indices = pruned_data[node_type].global_indices.cpu().tolist()
                pruned_data[node_type].entity_id = torch.arange(num_nodes, dtype=torch.long) 
                self.local_idx_to_entity_name[node_type] = [
                    self.reverse_entity_map[gidx] for gidx in pruned_global_indices
                ]
            self.hetero_data = pruned_data
            return pruned_data, self.local_idx_to_entity_name
        except Exception as e:
            print("Largest connected component pruning failed:", e)
            for node_type in data.node_types: 
                num_nodes = data[node_type].num_nodes
                data[node_type].node_id = torch.arange(num_nodes, dtype=torch.long)
                global_indices = data[node_type].global_indices.cpu().tolist()
                data[node_type].entity_id = torch.arange(num_nodes, dtype=torch.long)
                self.local_idx_to_entity_name[node_type] = [
                    self.reverse_entity_map[gidx] for gidx in global_indices
                ]
            self.hetero_data = data
            return data, self.local_idx_to_entity_name

    def split_datasets(self, n_splits=5, test_ratio=0.2, random_state=42):
        from collections import defaultdict
        import numpy as np
        print("\nSplitting dataset...")
        target_triples = []
        for target_type in TARGET_TYPES:  # ['Protein','TF','RBP']
            et = ("TCM_ID", "herb_modulates_target", target_type)
            if et in self.hetero_data.edge_types:
                ei = self.hetero_data[et].edge_index
                for src_idx, dst_idx in ei.t().tolist():
                    target_triples.append((src_idx, dst_idx, "herb_modulates_target", target_type))
        target_triples = np.array(target_triples, dtype=object)
        print(f"Herb-target triplet count: {len(target_triples)}")
        rng = np.random.RandomState(random_state)
        def group_by_herb(triples):
            herb2idx = defaultdict(list)
            for i, t in enumerate(triples):
                herb2idx[int(t[0])].append(i)  # t[0] = herb local idx
            herbs = sorted(herb2idx.keys())
            for h in herbs:
                herb2idx[h] = sorted(herb2idx[h])
            return herbs, herb2idx
        def make_test_cover_all_herbs(triples, test_ratio, rng):
            herbs, herb2idx = group_by_herb(triples)
            test_idx, remain_idx = [], []
            for h in herbs:
                idxs = np.array(herb2idx[h], dtype=int)
                if len(idxs) == 1:
                    remain_idx.extend(idxs.tolist())
                    continue
                n_test = max(1, int(round(len(idxs) * test_ratio)))
                n_test = min(n_test, len(idxs) - 1) 
                perm = rng.permutation(len(idxs))
                take = idxs[perm[:n_test]]
                left = idxs[perm[n_test:]]
                test_idx.extend(take.tolist())
                remain_idx.extend(left.tolist())
            return (np.array(sorted(remain_idx), dtype=int),
                    np.array(sorted(test_idx), dtype=int))
        t_remain_idx, t_test_idx = make_test_cover_all_herbs(target_triples, test_ratio, rng)
        t_remain = target_triples[t_remain_idx]
        t_test   = target_triples[t_test_idx]
        def edge_level_kfold_balanced(triples, n_splits, rng):
            triples = np.array(triples, dtype=object)
            herbs, herb2idx_all = group_by_herb(triples)
            htype2idx = defaultdict(list)
            for h in herbs:
                for i in herb2idx_all[h]:
                    ttype = triples[i][3]  # 'Protein'/'TF'/'RBP'
                    htype2idx[(h, ttype)].append(i)
            folds_val = [list() for _ in range(n_splits)]
            cursor = 0
            single_keep_train = set()
            for (h, ttype), idxs in htype2idx.items():
                if len(idxs) == 1:
                    single_keep_train.add(idxs[0])
            for (h, ttype), idxs in htype2idx.items():
                if len(idxs) <= 1:
                    continue
                perm = [idxs[p] for p in rng.permutation(len(idxs))]
                for j, idx in enumerate(perm):
                    folds_val[(cursor + j) % n_splits].append(int(idx))
                cursor = (cursor + len(perm)) % n_splits
            all_idx_set = set(range(len(triples)))
            ban_val = single_keep_train
            fold_pairs = []
            for i in range(n_splits):
                val_i = sorted(set(folds_val[i]) - ban_val)
                train_i = sorted(list(all_idx_set - set(val_i)))   # All remaining go to train
                fold_pairs.append((np.array(train_i, dtype=int), np.array(val_i, dtype=int)))
            return fold_pairs
        self.target_folds = edge_level_kfold_balanced(t_remain, n_splits, rng)
        self.t_remain = t_remain
        self.t_test   = t_test
        def report_folds(triples, folds):
            triples = np.array(triples, dtype=object)
            herbs, herb2idx = group_by_herb(triples)
            for fi, (tr_idx, va_idx) in enumerate(folds, 1):
                tr_set = set(tr_idx.tolist())
                va = [triples[i] for i in va_idx.tolist()]
                cold = 0
                ty_count = defaultdict(int)
                herb_train_edges = defaultdict(list)
                for i in tr_idx.tolist():
                    h = int(triples[i][0])
                    herb_train_edges[h].append(i)
                for i in va_idx.tolist():
                    h, _, _, ttype = triples[i]
                    ty_count[ttype] += 1
                    if len(herb_train_edges.get(int(h), [])) == 0:
                        cold += 1
                total_val = len(va_idx)
                parts = ", ".join([f"{k}:{ty_count[k]}" for k in TARGET_TYPES if ty_count.get(k, 0) > 0])
                print(f"[Fold{fi}] val_edges={total_val} | by_type {{{parts}}} | cold_herb={cold}")
                if cold > 0:
                    print(f"  -> Warning: Found {cold} cold-start herbs (no positive edges in training set for this fold). Please check if warm-start constraint is violated by external process.")
        print(f"Train/validation total: {len(self.t_remain)}, test set count: {len(self.t_test)}")
        report_folds(self.t_remain, self.target_folds)
        print("5-fold cross-validation split completed: test set covers all herbs; validation set satisfies warm-start; balanced rotation by (herb,type).")

    def _create_edge_masks(self, train_edges, val_edges, test_edges):
        for target_type in TARGET_TYPES:
            edge_type = ("TCM_ID", "herb_modulates_target", target_type)
            if edge_type not in self.hetero_data.edge_types:
                continue
            edge_index = self.hetero_data[edge_type].edge_index
            train_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            val_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            test_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            edge_to_idx = {}
            for idx, (src, dst) in enumerate(edge_index.t().tolist()):
                edge_to_idx[(src, dst)] = idx
            for src, dst, _, t_type in train_edges:
                if t_type == target_type:
                    if (src, dst) in edge_to_idx: 
                        idx = edge_to_idx[(src, dst)]
                        train_mask[idx] = True
            for src, dst, _, t_type in val_edges:
                if t_type == target_type:
                    if (src, dst) in edge_to_idx: 
                        idx = edge_to_idx[(src, dst)]
                        val_mask[idx] = True
            for src, dst, _, t_type in test_edges:
                if t_type == target_type:
                    if (src, dst) in edge_to_idx: 
                        idx = edge_to_idx[(src, dst)]
                        test_mask[idx] = True
            self.hetero_data[edge_type].train_mask = train_mask
            self.hetero_data[edge_type].val_mask = val_mask
            self.hetero_data[edge_type].test_mask = test_mask
    
    def prepare_node_features(self, herb_data_dict):
        print("\nPreparing node features...")
        if hasattr(self.hetero_data["TCM_ID"], 'global_indices'):
            globals_ = self.hetero_data["TCM_ID"].global_indices.tolist()
            herb_tcm_ids = [self.reverse_entity_map[g] for g in globals_]
        else:
            herb_tcm_ids = [self.reverse_entity_map[i] for i in self.hetero_data["TCM_ID"].node_id.tolist()]
        text_list, nes_list = [], []
        for tcm_id in herb_tcm_ids:
            rec = herb_data_dict.get(tcm_id, {})
            text_list.append(rec.get("text_feat"))
            nes_list.append(rec.get("nes_feat"))
        def _stack_or_zero(lst, dim_guess=128):
            dim = next((int(x.numel()) for x in lst if isinstance(x, torch.Tensor)), dim_guess)
            out = [(x if isinstance(x, torch.Tensor) else torch.zeros(dim)) for x in lst]
            return torch.stack(out).float()
        self.hetero_data["TCM_ID"].text_feat = _stack_or_zero(text_list)
        self.hetero_data["TCM_ID"].nes_feat  = _stack_or_zero(nes_list)
        print("Herb text_feat:", tuple(self.hetero_data["TCM_ID"].text_feat.shape),
            "nes_feat:", tuple(self.hetero_data["TCM_ID"].nes_feat.shape))

        