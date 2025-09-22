import os, ast, hashlib
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

herb_data_dict = {}   # { TCM_ID: {"text_feat": Tensor[ds], "nes_feat": Tensor[dn], "chinese_name": str} }
def load_herb_mapping(mapping_csv_path):
    mapping_df = pd.read_csv(mapping_csv_path)
    name_to_id = dict(zip(mapping_df["Chinese_name"], mapping_df["TCM_ID"]))
    id_to_name = dict(zip(mapping_df["TCM_ID"], mapping_df["Chinese_name"]))
    return name_to_id, id_to_name
def _row_to_sentence(name: str, row: pd.Series) -> str:
    natures   = [c for c in ['寒','热','温','凉','平'] if pd.notna(row.get(c,'')) and str(row.get(c,'')).strip()]
    tastes    = [c for c in ['酸','苦','甘','辛','咸']   if pd.notna(row.get(c,'')) and str(row.get(c,'')).strip()]
    meridians = [c for c in ['肺','心包','心','大肠','三焦','小肠','胃','胆','膀胱','脾','肝','肾']
                 if pd.notna(row.get(c,'')) and str(row.get(c,'')).strip()]
    tox_raw   = str(row.get('毒性','')).strip()
    tox_text  = tox_raw if tox_raw else '无毒'
    eff_text = ""
    eff_raw = row.get('功效', "")
    if pd.notna(eff_raw) and str(eff_raw).strip():
        try:
            lst = ast.literal_eval(str(eff_raw))
            if isinstance(lst, (list, tuple)):
                eff_text = "、".join([str(x).strip() for x in lst if str(x).strip()])
            else:
                eff_text = str(eff_raw).strip()
        except Exception:
            eff_text = str(eff_raw).strip()
    parts = []
    if natures:   parts.append("性" + "、".join(natures))
    if tastes:    parts.append("味" + "、".join(tastes))
    if meridians: parts.append("归" + "、".join(meridians) + "经")
    parts.append(tox_text)
    if eff_text:  parts.append("功效：" + eff_text)
    return f"{name}；" + "；".join(parts)

def _embed_sentences(sentences, model_name, device=None, batch_size=64, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    if SentenceTransformer is None:
        raise ImportError("pip install sentence-transformers")
    if not os.path.isdir(model_name):
        raise FileNotFoundError(f"本地模型目录不存在: {model_name}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    vecs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="文本向量编码", unit="batch"):
        batch = sentences[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(emb)
    X = np.vstack(vecs).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, X)
    return X

def _load_nes(nes_csv_path, herb_order):
    nes = pd.read_csv(nes_csv_path, index_col=0).T
    nes.index = nes.index.str.strip()
    nes = nes.reindex(herb_order)
    nes = nes.fillna(0.0)
    X = nes.values.astype(np.float32)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X).astype(np.float32)
    return Xz

def build_text_nes_features(property_csv_path, nes_csv_path, mapping_csv_path,
                            text_model, cache_dir=None, device=None):
    global herb_data_dict
    name_to_id, _ = load_herb_mapping(mapping_csv_path)
    prop = pd.read_csv(property_csv_path, index_col=0)
    prop.index = prop.index.astype(str).str.strip()
    nes_all = pd.read_csv(nes_csv_path, index_col=0).T
    nes_all.index = nes_all.index.astype(str).str.strip()
    herbs = sorted(set(prop.index) & set(nes_all.index))
    if not herbs:
        raise ValueError("属性表与 NES 没有交集中药，请检查名称一致。")
    prop = prop.loc[herbs]
    sentences = [_row_to_sentence(h, prop.loc[h]) for h in herbs]
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tok = hashlib.md5((str(text_model) + str(len(sentences))).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"sent_emb_{tok}.npy")
    S = _embed_sentences(sentences, model_name=text_model, device=device, cache_path=cache_path)  # [N, ds]
    ds = int(S.shape[1])
    N = _load_nes(nes_csv_path, herb_order=herbs).astype(np.float32)  # [N, dn]
    dn = int(N.shape[1])
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12).astype(np.float32)
    herb_data_dict.clear()
    for i, name in enumerate(herbs):
        tcm_id = name_to_id.get(name, f"UNKNOWN:{name}")
        herb_data_dict[tcm_id] = {
            "text_feat": torch.from_numpy(S[i]).float(),  # [ds]
            "nes_feat":  torch.from_numpy(N[i]).float(),  # [dn]
            "chinese_name": name
        }
    return ds, dn