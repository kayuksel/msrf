"""Leaf-RFM readout head, vendored for self-containment.

Implementation of the leaf regime of xRFM (Beaglehole et al., "xRFM: accurate, scalable and
interpretable feature learning models for tabular data", ICLR 2026): kernel ridge with the
L_p^q kernel K(u,v) = exp(-||u-v||_p^q / L^q), iterated diag- or full-matrix AGOP (Mahalanobis)
reweighting, adaptive median-heuristic bandwidth, a multi-lambda sweep via one eigendecomposition
per iteration, and per-(config, iteration, lambda) selection on stratified OOF. Closed-form
throughout; AGOP gradients are analytic with the self-term skipped (their App. A.4).

Used here only as a CEILING ABLATION -- it is not the deployed readout, and it forfeits the
exact attribution, rank-one personalisation and kilobyte footprint that the linear head provides.
"""
import math

import numpy as np
import torch

_TEMPS = (0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 6.0)
_MAX_AGOP_CLASSES = 12
ALPHAS = np.logspace(-3, 3, 13)


def _torch_ridge(phi_s, Y, lam):
    """Closed-form ridge W = (Phi'Phi + lam I)^-1 Phi'Y on standardized phi columns."""
    K = phi_s.shape[1]
    eye = torch.eye(K, dtype=phi_s.dtype, device=phi_s.device)
    A = phi_s.T @ phi_s + lam * eye
    B = phi_s.T @ Y
    try:
        Wt = torch.linalg.solve(A, B)
        if not torch.isfinite(Wt).all():
            raise RuntimeError
    except Exception:
        Wt = torch.linalg.lstsq(A + (10.0 * lam + 1.0) * eye, B).solution
    return Wt



def make_folds(n, y_cls, n_splits=3, seed=0):
    """List of (fit_idx, val_idx). Stratified when y_cls is given (each class in every fold if
    possible); reduces fold count for tiny classes; returns [] if no valid split exists."""
    from sklearn.model_selection import StratifiedKFold, KFold
    if y_cls is None:
        ns = min(n_splits, n)
        if ns < 2:
            return []
        return list(KFold(ns, shuffle=True, random_state=seed).split(np.zeros(n)))
    cc = np.bincount(y_cls)
    ns = min(n_splits, int(cc[cc > 0].min()))
    if ns < 2 or (cc > 0).sum() < 2:
        return []
    return list(StratifiedKFold(ns, shuffle=True, random_state=seed).split(np.zeros(n), y_cls))


def _softmax_np(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def _clf_oof_metric(y, S, C):
    """Task-aligned OOF metric from raw scores S (n,C): binary -> 1-AUC; multi -> temp-tuned log_loss."""
    from sklearn.metrics import roc_auc_score, log_loss
    if C == 2:
        try:
            return 1.0 - roc_auc_score(y, S[:, 1] - S[:, 0])
        except ValueError:
            return np.inf
    best = np.inf
    for t in _TEMPS:
        v = log_loss(y, np.clip(_softmax_np(S / t), 1e-12, None), labels=list(range(C)))
        best = min(best, v)
    return best


def tune_temperature(y, S, C):
    """Best softmax temperature by OOF log_loss (rank-preserving; matches champion._calibrate)."""
    from sklearn.metrics import log_loss
    best, T = np.inf, 1.0
    for t in _TEMPS:
        v = log_loss(y, np.clip(_softmax_np(S / t), 1e-12, None), labels=list(range(C)))
        if v < best:
            best, T = v, t
    return T


# --------------------------------------------------------------------------- 1) AGOP channel reweighting

# --------------------------------------------------------------------------- 2) leaf RFM (L_p^q kernel + AGOP)
class RFMHead:
    """One leaf RFM (paper Alg. A.1): kernel K(u,v) = exp(-||u-v||_p^q / L^q) on Mahalanobis-scaled
    inputs, T iterations of AGOP reweighting (diag or full), lambda swept via one eigendecomposition
    per iteration, per-(config, iteration, lambda) selection on stratified OOF; the winner is refit
    on the full support for the query prediction. The M trajectory is always built at a fixed
    reference lambda (lam_agop) so the searched and refit trajectories match; the bandwidth is
    re-adapted (median heuristic) after every M update, which also absorbs AGOP normalization."""

    def __init__(self, configs=None, n_iters=3, lam_grid=None, seed=0, agop_eps=1e-3):
        self.configs = configs or [
            dict(p=1.0, q=1.0, bw=0.5, diag=True), dict(p=1.0, q=1.0, bw=1.0, diag=True),
            dict(p=1.0, q=1.0, bw=2.0, diag=True), dict(p=1.0, q=1.0, bw=1.0, diag=False),
            dict(p=2.0, q=1.0, bw=0.5, diag=True), dict(p=2.0, q=1.0, bw=1.0, diag=True),
            dict(p=2.0, q=1.0, bw=2.0, diag=True), dict(p=2.0, q=1.0, bw=1.0, diag=False),
        ]
        self.n_iters = n_iters
        self.lam_grid = lam_grid or [10.0 ** e for e in (-4, -3, -2, -1, 0, 1)]
        self.lam_agop = self.lam_grid[len(self.lam_grid) // 2]
        self.seed = seed
        self.agop_eps = agop_eps

    # --- kernel machinery -------------------------------------------------
    @staticmethod
    def _apply_M(X, M_state):
        kind, M = M_state
        if kind == 'none':
            return X
        if kind == 'diag':
            return X * M.unsqueeze(0)                          # M = diag(M_mat)^{1/2}, a (d,) vector
        return X @ M                                           # M = M_mat^{1/2}, symmetric (d,d)

    def _grads_at(self, Xq_M, Xs_M, alpha, cfg, L, skip_self=False):
        """Analytic per-sample gradients of f_k(z) = sum_i alpha_ik K(z, x_i) at rows of Xq_M,
        w.r.t. the M-SCALED coordinates. Returns (B, C, d). skip_self masks the i==j self-term
        (paper App. A.4: the kernel may be non-differentiable at u=v)."""
        p, q = cfg['p'], cfg['q']
        Dq = torch.cdist(Xq_M, Xs_M, p=p).clamp(min=0)
        Kq = torch.exp(-(Dq / L) ** q)
        B, n = Kq.shape
        d = Xs_M.shape[1]
        out = torch.zeros(B, alpha.shape[1], d)
        bs = max(1, int(2e7 // max(n * d, 1)))
        for i0 in range(0, B, bs):
            i1 = min(B, i0 + bs)
            diff = Xq_M[i0:i1].unsqueeze(1) - Xs_M.unsqueeze(0)          # (b, n, d)
            Dsafe = Dq[i0:i1].clamp(min=1e-9)
            coef = -Kq[i0:i1] * (q / (L ** q)) * Dsafe ** (q - p)        # (b, n)
            if skip_self:
                coef = coef * (Dq[i0:i1] >= 1e-9)
            G = coef.unsqueeze(2) * torch.sign(diff) * diff.abs().clamp(min=1e-12) ** (p - 1.0)
            out[i0:i1] = torch.einsum('bnd,nc->bcd', G, alpha)
        return out

    def _agop(self, Xs_M, alpha, cfg, L):
        """AGOP over support points, w.r.t. the M-scaled coords (self-term skipped).
        diag -> (d,), full -> (d,d)."""
        if alpha.shape[1] > _MAX_AGOP_CLASSES:                # hi-class: random output projection
            g = torch.Generator().manual_seed(777 + self.seed)
            R = torch.randn(alpha.shape[1], _MAX_AGOP_CLASSES, generator=g) / math.sqrt(_MAX_AGOP_CLASSES)
            alpha = alpha @ R
        n, d = Xs_M.shape
        diag_acc = torch.zeros(d)
        full_acc = None if cfg['diag'] else torch.zeros(d, d)
        bs = max(1, int(2e7 // max(n * d, 1)))
        for i0 in range(0, n, bs):
            G = self._grads_at(Xs_M[i0:min(n, i0 + bs)], Xs_M, alpha, cfg, L, skip_self=True)
            diag_acc += (G ** 2).sum((0, 1))
            if full_acc is not None:
                full_acc += torch.einsum('bcd,bce->de', G, G)
        return (diag_acc / n) if cfg['diag'] else (full_acc / n)

    def _to_original_coords(self, A, M_state, diag):
        """Gradients were taken w.r.t. scaled coords; convert AGOP to ORIGINAL coordinates
        (paper Alg. A.1 recomputes M w.r.t. the original x each iteration)."""
        kind, M = M_state
        if kind == 'none':
            return A
        if diag:                                               # kind == 'diag' by construction
            return A * M ** 2
        return M @ A @ M
    def _M_from_agop(self, A, cfg):
        """Normalize AGOP, return the M^c operator state applied to ORIGINAL inputs next iter.
        cfg['agop_c'] is the paper's matrix-power c (default 1/2; they also use 1/4)."""
        c = float(cfg.get('agop_c', 0.5))
        if cfg['diag']:
            m = A / (A.max().clamp(min=1e-12) + self.agop_eps)
            return ('diag', (m + self.agop_eps) ** c)
        A = A / (A.abs().max().clamp(min=1e-12) + self.agop_eps)
        A = A + self.agop_eps * torch.eye(A.shape[0])
        w, V = torch.linalg.eigh(A)
        return ('full', (V * w.clamp(min=0) ** c) @ V.T)

    def _solve_all_lams(self, Ks, Y, lams):
        """alpha(lam) for every lam via one eigh. Returns dict lam -> (n, C)."""
        w, V = torch.linalg.eigh(Ks + 1e-8 * torch.eye(Ks.shape[0]))
        w = w.clamp(min=0)
        VtY = V.T @ Y
        return {lam: V @ (VtY / (w + lam).unsqueeze(1)) for lam in lams}

    @staticmethod
    def _median_bw(D):
        off = D[~torch.eye(D.shape[0], dtype=bool)]
        med = float(off.median()) if off.numel() else 1.0
        return max(med, 1e-3)

    # --- one config: iterate M, score all (t, lam) on Xev -------------------
    def _run_config(self, Xtr, Ytr, Xev, cfg, t_stop=None, lam_only=None):
        """If t_stop is None: return {(t, lam): scores on Xev} for the whole trajectory.
        Else: run exactly t_stop iterations and return scores for lam_only (the refit path)."""
        M_state = ('none', None)
        results = {}
        n_iters = self.n_iters if t_stop is None else (t_stop + 1)
        for t in range(n_iters):
            Xtr_M = self._apply_M(Xtr, M_state)
            Xev_M = self._apply_M(Xev, M_state)
            Dtr = torch.cdist(Xtr_M, Xtr_M, p=cfg['p']).clamp(min=0)
            L = cfg['bw'] * self._median_bw(Dtr)
            Ktr = torch.exp(-(Dtr / L) ** cfg['q'])
            Kev = torch.exp(-(torch.cdist(Xev_M, Xtr_M, p=cfg['p']).clamp(min=0) / L) ** cfg['q'])
            need = self.lam_grid if t_stop is None else sorted({self.lam_agop, lam_only})
            alphas = self._solve_all_lams(Ktr, Ytr, need)
            if t_stop is None:
                for lam, a in alphas.items():
                    results[(t, lam)] = (Kev @ a).numpy()
            elif t == t_stop:
                return (Kev @ alphas[lam_only]).numpy()
            if t < n_iters - 1:
                A = self._agop(Xtr_M, alphas[self.lam_agop], cfg, L)
                A = self._to_original_coords(A, M_state, cfg['diag'])
                M_state = self._M_from_agop(A, cfg)
        return results

    # --- public ------------------------------------------------------------
    def fit_predict(self, Xs, y_s, Xq, is_reg, n_classes, folds_seed=0):
        """Stratified-OOF search over (config, t, lam) -> full-support refit of the winner -> query
        scores. Returns (query_scores (nq, C), info) where info carries the winner's OOF scores
        (for temperature calibration) and the chosen hyperparameters."""
        Xs = torch.as_tensor(np.nan_to_num(np.asarray(Xs, np.float32)), dtype=torch.float32)
        Xq = torch.as_tensor(np.nan_to_num(np.asarray(Xq, np.float32)), dtype=torch.float32)
        ns = Xs.shape[0]
        y_np = None if is_reg else np.asarray(y_s, dtype=int)
        if is_reg:
            y_t = torch.as_tensor(np.asarray(y_s), dtype=torch.float32)
            mu_y, sd_y = y_t.mean(), y_t.std().clamp(min=1e-6)
            Y = ((y_t - mu_y) / sd_y).unsqueeze(1)
        else:
            Y = torch.nn.functional.one_hot(torch.as_tensor(y_np).long(), n_classes).float()
        folds = make_folds(ns, y_np, 3, folds_seed)
        best = None                                            # (E, cfg_i, t, lam)
        best_oof, best_ok = None, None
        for ci, cfg in enumerate(self.configs):
            if not folds:
                break
            oof, ok = {}, np.zeros(ns, dtype=bool)
            for fit_idx, val_idx in folds:
                if (not is_reg) and len(np.unique(y_np[fit_idx])) < n_classes:
                    continue
                res = self._run_config(Xs[fit_idx], Y[fit_idx], Xs[val_idx], cfg)
                for key, S in res.items():
                    oof.setdefault(key, np.full((ns, Y.shape[1]), np.nan))[val_idx] = S
                ok[val_idx] = True
            if ok.sum() < 2 or (not is_reg and len(np.unique(y_np[ok])) < 2):
                continue
            for (t, lam), S in sorted(oof.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                if np.isnan(S[ok]).any():
                    continue
                if is_reg:
                    E = float(np.sqrt(np.mean((S[ok, 0] - Y[ok, 0].numpy()) ** 2)))
                else:
                    E = _clf_oof_metric(y_np[ok], S[ok], n_classes)
                if best is None or E < best[0]:
                    best = (E, ci, t, lam)
                    best_oof, best_ok = S, ok
        if best is None:                                       # no valid folds -> defaults, no iteration
            best = (np.inf, 1, 0, self.lam_agop)
        E, ci, t_star, lam_star = best
        cfg = self.configs[ci]
        Sq = self._run_config(Xs, Y, Xq, cfg, t_stop=t_star, lam_only=lam_star)
        if is_reg:
            Sq = Sq * float(sd_y) + float(mu_y)
        info = {'oof_E': float(E), 'config': dict(cfg), 't': int(t_star), 'lam': float(lam_star),
                'oof_scores': best_oof, 'oof_mask': best_ok}
        return Sq, info
