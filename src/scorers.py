import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseScorer(nn.Module):
    pairwise: bool = False 

class ConcatMLPScorer(BaseScorer):
    pairwise = False
    def __init__(self, dim: int, hidden=(512, 256), dropout=0.0, inner_cols: int = None):
        super().__init__()
        self.dim = dim
        if isinstance(hidden, int):
            hidden = (hidden,)
        assert len(hidden) >= 1, "hidden 至少包含一层宽度"
        h0 = hidden[0]
        self.fh = nn.Linear(dim, h0, bias=False)   # W_h
        self.ft = nn.Linear(dim, h0, bias=False)   # W_t
        self.b1 = nn.Parameter(torch.zeros(h0))    # b
        tails = []
        in_dim = h0
        for h in hidden[1:]:
            tails += [nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, h)]
            in_dim = h
        tails += [nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, 1)]
        self.tail = nn.Sequential(*tails)
        self.inner_cols = inner_cols
    def forward(self, *args):
        if len(args) == 1:
            pair_2d = args[0]
            d = self.dim
            h = pair_2d[..., :d]
            t = pair_2d[..., d:]
            x = self.fh(h) + self.ft(t) + self.b1         # [K, h0]
            x = F.gelu(x)
            y = self.tail(x).squeeze(-1)                  # [K]
            return y
        elif len(args) == 2:
            src, dst = args
            return self._score_block(src, dst)
        else:
            raise TypeError("ConcatMLPScorer.forward 期望 1 或 2 个参数")

    def _score_block(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        bs = src.size(0)
        nd = dst.size(0)
        h0 = self.fh.out_features
        H1 = self.fh(src)  # [bs, h0]
        outputs = []
        cols = self.inner_cols
        if cols is None or cols <= 0:
            cols = max(1, 200_000 // max(1, bs))
        for j in range(0, nd, cols):
            Tc = dst[j:j+cols]                 # [cols, d]
            T1 = self.ft(Tc)                   # [cols, h0]
            x = H1[:, None, :] + T1[None, :, :] + self.b1  # [bs, cols, h0]
            x = F.gelu(x)
            x2 = x.reshape(-1, h0)             # [(bs*cols), h0]
            y2 = self.tail(x2).reshape(bs, -1) # [bs, cols]
            outputs.append(y2)
        return torch.cat(outputs, dim=1)        # [bs, nd]

class TwoTowerScorer(BaseScorer):
    def __init__(self, dim: int, hidden=512, out_dim=None, dropout=0.0):
        super().__init__()
        h = hidden; o = out_dim or hidden
        self.fh = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))
        self.ft = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))
    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.fh(src) @ self.ft(dst).t()  # [B, N]

class DotScorer(BaseScorer):
    def forward(self, src, dst): return src @ dst.t()

class BilinearScorer(BaseScorer):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)
    def forward(self, src, dst): return (src @ self.W) @ dst.t()

class DistMultScorer(BaseScorer):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.w, a=-0.1, b=0.1)
    def forward(self, src, dst): return (src * self.w) @ dst.t()

SCORER_REGISTRY = {
    "Concat_MLP": ConcatMLPScorer, 
    "Two_Tower":  TwoTowerScorer,
    "Dot":        DotScorer,
    "Bilinear":   BilinearScorer,
    "DistMultScorer":   DistMultScorer,
}

def build_scorer(name, dim=None, **kwargs):
    import inspect
    key = str(name).strip().lower()
    registry = {k.lower(): v for k, v in SCORER_REGISTRY.items()}
    if key not in registry:
        raise ValueError(f"Unknown scorer: {name}. Options={list(SCORER_REGISTRY.keys())}")
    cls = registry[key]
    sig = inspect.signature(cls)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    ctor_kwargs = {}
    if accepts_var_kw:
        ctor_kwargs = dict(kwargs)
    else:
        ctor_kwargs = {k: v for k, v in kwargs.items() if k in params}
    if dim is not None:
        for candidate in ("dim", "d", "in_dim", "input_dim", "embed_dim", "hidden_dim"):
            if candidate in params:
                ctor_kwargs[candidate] = dim
                break
    try:
        return cls(**ctor_kwargs)
    except TypeError:
        return cls()

