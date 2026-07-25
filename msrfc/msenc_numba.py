"""Numba (njit) port of the windowed multi-space encoder phi.

All 19 families are evaluated for one length-L patch inside a single ``@njit`` function ``_phi_one``; the
collection is processed by ``_phi_batch`` with ``prange`` over patches (parallel). numba does not support
``np.fft``, so the 3 FFT families are expressed as precomputed DFT matrices (cos/sin banks) -> the whole
kernel is matmul + scalar loops, fully nopython. Verified numerically identical to the numpy reference
(msenc_np.py), which is in turn identical to the torch reference implementation.

Constants (DFT banks, normalised Gaussian windows, conv & Ricker kernels, srf banks) are built once in numpy
in ``build_consts`` and passed into the njit kernels.
"""
import math
import numpy as np
from numba import njit, prange

L = 64
_EPS = 1e-8


# ---------- constant banks (numpy, built once) ----------
def build_consts(banks=None):
    if banks is None:
        from ._banks_data import load_banks
        banks = load_banks()
    trf_mu = np.asarray(banks["trf_mu"], np.float64)
    trf_sig = np.asarray(banks["trf_sig"], np.float64)
    t = np.arange(L, dtype=np.float64) / L
    win = np.exp(-((t[None, :] - trf_mu[:, None]) ** 2) / (2 * trf_sig[:, None] ** 2 + _EPS))
    GW = win / (win.sum(1, keepdims=True) + _EPS)                       # (12, L) normalised windows

    nn = np.arange(L, dtype=np.float64)
    kr = np.arange(L // 2 + 1, dtype=np.float64)
    a = 2 * np.pi * np.outer(kr, nn) / L
    Cr, Ci = np.cos(a), np.sin(a)                                       # rfft cos/sin (33, L)
    kf = np.arange(L, dtype=np.float64)
    af = 2 * np.pi * np.outer(kf, nn) / L
    Fc, Fs = np.cos(af), np.sin(af)                                     # full DFT cos/sin (L, L)
    hfir = np.where(kf == 0, 1.0, np.where(kf < L / 2, 2.0, np.where(kf == L // 2, 1.0, 0.0)))

    crf = np.ascontiguousarray(np.asarray(banks["crf_w"], np.float64)[:, 0, :])   # (16, 9)
    HYW = np.ascontiguousarray(np.asarray(banks["hydra_w"], np.float64))           # (2, 4, 9)
    x = np.linspace(-7.0, 7.0, 15)
    Rk = np.empty((4, 15), np.float64)
    for i, s in enumerate((1.5, 2.5, 4.0, 6.0)):
        r = (1.0 - (x / s) ** 2) * np.exp(-(x ** 2) / (2 * s * s))
        Rk[i] = (r - r.mean()) / (np.abs(r).sum() + _EPS)
    srfW = np.ascontiguousarray(np.asarray(banks["srf_W"], np.float64))           # (12,6,12)
    srfb = np.ascontiguousarray(np.asarray(banks["srf_b"], np.float64))           # (12,6)
    srfu = np.ascontiguousarray(np.asarray(banks["srf_u"], np.float64))           # (12,6)
    return (GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW)


# ---------- njit helpers ----------
@njit(cache=True)
def _quantile_lin(sorted_x, q):
    n = sorted_x.shape[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_x[n - 1]
    return sorted_x[lo] + frac * (sorted_x[lo + 1] - sorted_x[lo])


@njit(cache=True)
def _conv_same(w, ker, dilation):
    """cross-correlation, 'same' padding = dilation*(k//2); length L."""
    Lw = w.shape[0]
    k = ker.shape[0]
    pad = dilation * (k // 2)
    out = np.zeros(Lw, np.float64)
    for t in range(Lw):                       # out_len == L for these configs
        acc = 0.0
        for j in range(k):
            idx = t + j * dilation - pad
            if 0 <= idx < Lw:
                acc += ker[j] * w[idx]
        out[t] = acc
    return out


@njit(cache=True)
def _phi_one(w, GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW):
    """One length-L patch -> 173 features, filled in family order."""
    n = w.shape[0]
    out = np.empty(173, np.float64)
    o = 0

    mean = 0.0
    for i in range(n):
        mean += w[i]
    mean /= n
    c = w - mean
    ss = 0.0
    for i in range(n):
        ss += c[i] * c[i]
    var1 = ss / (n - 1)                              # unbiased (ddof=1), matches torch
    std1 = math.sqrt(var1)
    sden = std1 if std1 > _EPS else _EPS

    sw = np.sort(w)
    q25 = _quantile_lin(sw, 0.25); q50 = _quantile_lin(sw, 0.50); q75 = _quantile_lin(sw, 0.75)
    dw = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        dw[i] = w[i + 1] - w[i]

    # ===== family 0: stats (12) =====
    skew = 0.0; kurt = 0.0
    for i in range(n):
        skew += c[i] ** 3; kurt += c[i] ** 4
    skew = (skew / n) / (sden ** 3 + _EPS)
    kurt = (kurt / n) / (sden ** 4 + _EPS) - 3.0
    a1n = 0.0; d0 = 0.0; d1 = 0.0
    for i in range(n - 1):
        a1n += c[i] * c[i + 1]; d0 += c[i] * c[i]; d1 += c[i + 1] * c[i + 1]
    ac1 = a1n / (math.sqrt(d0) * math.sqrt(d1) + _EPS)
    absdiff = 0.0
    for i in range(n - 1):
        absdiff += abs(dw[i])
    absdiff /= (n - 1)
    st = np.empty(12, np.float64)
    st[0] = mean; st[1] = sden; st[2] = skew; st[3] = kurt
    st[4] = q25; st[5] = q50; st[6] = q75; st[7] = q75 - q25
    st[8] = ac1; st[9] = absdiff; st[10] = sw[0]; st[11] = sw[n - 1]
    for i in range(12):
        out[o] = st[i]; o += 1

    # ===== family 1: srf_mlp (12) =====
    for kk in range(12):
        acc = 0.0
        for dd in range(6):
            h = srfb[kk, dd]
            for mm in range(12):
                h += st[mm] * srfW[kk, dd, mm]
            if h < 0.0:
                h = 0.0
            acc += h * srfu[kk, dd]
        out[o] = acc; o += 1

    # ===== family 2: autocorr lags 1,2 (2) =====
    a2n = 0.0; a2d = 0.0
    for i in range(n - 2):
        a2n += c[i] * c[i + 2]; a2d += c[i] * c[i]
    out[o] = a1n / (d0 + _EPS); o += 1
    out[o] = a2n / (a2d + _EPS); o += 1

    # ----- rfft magnitude (shared by spectral & fftbands) -----
    nb = Cr.shape[0]                                 # 33
    mag = np.empty(nb, np.float64)
    for kb in range(nb):
        re = 0.0; im = 0.0
        for i in range(n):
            re += w[i] * Cr[kb, i]; im -= w[i] * Ci[kb, i]
        mag[kb] = math.sqrt(re * re + im * im)

    # ===== family 3: spectral centroid + entropy (2) =====
    psum = 0.0
    for kb in range(1, nb):
        psum += mag[kb]
    psum += _EPS
    cent = 0.0; ent = 0.0
    for j in range(nb - 1):
        pj = mag[j + 1] / psum
        cent += j * pj; ent -= pj * math.log(pj + 1e-12)
    out[o] = cent; o += 1
    out[o] = ent; o += 1

    # ===== family 4: turning (2) =====
    tp = 0.0
    for i in range(n - 2):
        if dw[i + 1] * dw[i] < 0.0:
            tp += 1.0
    fpos = 0.0
    for i in range(n - 1):
        if dw[i] > 0.0:
            fpos += 1.0
    out[o] = tp / (n - 2); o += 1
    out[o] = fpos / (n - 1); o += 1

    # ===== family 5: trf_gausswin (12) = w @ GW.T =====
    for kk in range(12):
        acc = 0.0
        for i in range(n):
            acc += w[i] * GW[kk, i]
        out[o] = acc; o += 1

    # conv outputs at dilations 2,4 (cached for ppv/max/position)
    conv2 = np.empty((16, n), np.float64)
    conv4 = np.empty((16, n), np.float64)
    for ci in range(16):
        conv2[ci] = _conv_same(w, crf[ci], 2)
        conv4[ci] = _conv_same(w, crf[ci], 4)

    # ===== family 6: crf_ppv (32) = ppv@dil2 (16) then ppv@dil4 (16) =====
    for ci in range(16):
        p = 0.0
        for t in range(n):
            if conv2[ci, t] > 0.0:
                p += 1.0
        out[o] = p / n; o += 1
    for ci in range(16):
        p = 0.0
        for t in range(n):
            if conv4[ci, t] > 0.0:
                p += 1.0
        out[o] = p / n; o += 1

    # ===== family 7: hilbert_env (2) =====
    yre = np.empty(L, np.float64); yim = np.empty(L, np.float64)
    for kb in range(L):
        re = 0.0; im = 0.0
        for i in range(n):
            re += w[i] * Fc[kb, i]; im -= w[i] * Fs[kb, i]
        yre[kb] = re * hfir[kb]; yim[kb] = im * hfir[kb]
    emean = 0.0
    env = np.empty(L, np.float64)
    for ni in range(L):
        ir = 0.0; ii = 0.0
        for kb in range(L):
            ir += yre[kb] * Fc[kb, ni] - yim[kb] * Fs[kb, ni]
            ii += yre[kb] * Fs[kb, ni] + yim[kb] * Fc[kb, ni]
        ir /= L; ii /= L
        env[ni] = math.sqrt(ir * ir + ii * ii); emean += env[ni]
    emean /= L
    evar = 0.0; eabs = 0.0
    for i in range(L):
        evar += (env[i] - emean) ** 2
    for i in range(L - 1):
        eabs += abs(env[i + 1] - env[i])
    out[o] = math.sqrt(evar / (L - 1)) / (emean + _EPS); o += 1
    out[o] = (eabs / (L - 1)) / (emean + _EPS); o += 1

    # ===== family 8: crf_max (32) = max@dil2 (16) then max@dil4 (16) =====
    for ci in range(16):
        m = conv2[ci, 0]
        for t in range(1, n):
            if conv2[ci, t] > m:
                m = conv2[ci, t]
        out[o] = m; o += 1
    for ci in range(16):
        m = conv4[ci, 0]
        for t in range(1, n):
            if conv4[ci, t] > m:
                m = conv4[ci, t]
        out[o] = m; o += 1

    # ===== family 9: morphology_updown (4) =====
    pmax = 0.0; nmax = 0.0; pe = 0.0; ne = 0.0
    for i in range(n - 1):
        d = dw[i]
        if d > 0.0:
            if d > pmax:
                pmax = d
            pe += d * d
        elif d < 0.0:
            if -d > nmax:
                nmax = -d
            ne += d * d
    out[o] = pmax; o += 1
    out[o] = nmax; o += 1
    out[o] = math.log((pmax + 1e-6) / (nmax + 1e-6)); o += 1
    out[o] = math.log((pe + 1e-6) / (ne + 1e-6)); o += 1

    # ===== family 10: fftbands (6) =====
    s2 = 0.0
    pw = np.empty(nb - 1, np.float64)
    for j in range(nb - 1):
        pw[j] = mag[j + 1] * mag[j + 1]; s2 += pw[j]
    s2 += _EPS
    for b0 in range(0, nb - 1, 6):
        b1 = b0 + 6
        if b1 > nb - 1:
            b1 = nb - 1
        acc = 0.0
        for j in range(b0, b1):
            acc += pw[j] / s2
        if acc < 1e-8:
            acc = 1e-8
        out[o] = math.log(acc); o += 1

    # ===== family 11: perm_entropy + Hjorth mobility/complexity (3) =====
    H = np.zeros(8, np.float64)
    for i in range(n - 2):
        a = w[i]; b = w[i + 1]; cc = w[i + 2]
        code = 0
        if a < b:
            code += 4
        if b < cc:
            code += 2
        if a < cc:
            code += 1
        H[code] += 1.0
    pent = 0.0
    for k in range(8):
        Hk = H[k] / (n - 2)
        pent -= Hk * math.log(Hk + 1e-12)
    pent /= math.log(6.0)
    # var1(d1) and var1(d2), ddof=1
    md = 0.0
    for i in range(n - 1):
        md += dw[i]
    md /= (n - 1)
    vd1 = 0.0
    for i in range(n - 1):
        vd1 += (dw[i] - md) ** 2
    vd1 /= (n - 2)
    m2d = 0.0
    d2 = np.empty(n - 2, np.float64)
    for i in range(n - 2):
        d2[i] = w[i + 2] - 2.0 * w[i + 1] + w[i]; m2d += d2[i]
    m2d /= (n - 2)
    vd2 = 0.0
    for i in range(n - 2):
        vd2 += (d2[i] - m2d) ** 2
    vd2 /= (n - 3)
    mob = math.sqrt((vd1 + _EPS) / (var1 + _EPS))
    comp = math.sqrt((vd2 + _EPS) / (vd1 + _EPS)) / (mob + _EPS)
    out[o] = pent; o += 1
    out[o] = mob; o += 1
    out[o] = comp; o += 1

    # ===== family 12: curvature (3) =====
    cabsm = 0.0; cabsx = 0.0; cpe = 0.0; cne = 0.0
    for i in range(n - 2):
        cc = d2[i]; ac = abs(cc)
        cabsm += ac
        if ac > cabsx:
            cabsx = ac
        if cc > 0.0:
            cpe += cc * cc
        elif cc < 0.0:
            cne += cc * cc
    out[o] = cabsm / (n - 2); o += 1
    out[o] = cabsx; o += 1
    out[o] = math.log((cpe + 1e-6) / (cne + 1e-6)); o += 1

    # ===== family 13: conv_position (3) from dil2 argmax over 16 kernels =====
    pmean = 0.0
    posv = np.empty(16, np.float64)
    for ci in range(16):
        m = conv2[ci, 0]; am = 0
        for t in range(1, n):
            if conv2[ci, t] > m:
                m = conv2[ci, t]; am = t
        posv[ci] = am / n; pmean += posv[ci]
    pmean /= 16
    pvar = 0.0; pmn = posv[0]; pmx = posv[0]
    for ci in range(16):
        pvar += (posv[ci] - pmean) ** 2
        if posv[ci] < pmn:
            pmn = posv[ci]
        if posv[ci] > pmx:
            pmx = posv[ci]
    out[o] = pmean; o += 1
    out[o] = math.sqrt(pvar / 15); o += 1                 # std ddof=1 over 16 positions
    out[o] = pmx - pmn; o += 1

    # ===== family 14: ar_residual AR(2)+AR(3) (4) =====
    out[o] = 0.0; out[o + 1] = 0.0; out[o + 2] = 0.0; out[o + 3] = 0.0
    _ar_fit(c, 2, out, o)
    _ar_fit(c, 3, out, o + 2)
    o += 4

    # ===== family 15: acf_first_min (3) =====
    cc2 = 0.0
    for i in range(n):
        cc2 += c[i] * c[i]
    acf = np.empty(32, np.float64)
    for k in range(1, 33):
        s = 0.0
        for i in range(n - k):
            s += c[i] * c[i + k]
        acf[k - 1] = s / (cc2 + _EPS)
    has = False; fm = 0
    for j in range(1, 31):
        if acf[j] < acf[j - 1] and acf[j] <= acf[j + 1]:
            has = True; fm = j - 1            # argmax of ismin (first True), index into length-30
            break
    first_min = ((fm + 2.0) if has else 32.0) / n
    fz = 0
    for j in range(32):
        if acf[j] < 0.0:
            fz = j; break
    first_zero = fz / n
    idx_va = fm + 1
    if idx_va > 31:
        idx_va = 31
    out[o] = first_min; o += 1
    out[o] = first_zero; o += 1
    out[o] = acf[idx_va]; o += 1

    # ===== family 16: histogram_mode (3) =====
    zden = std1 if std1 > 1e-6 else 1e-6
    z = np.empty(n, np.float64)
    for i in range(n):
        z[i] = (w[i] - mean) / zden
    inv = 1.0 / (5.0 / 9.0)
    e = np.empty(10, np.float64)
    maxl = -1e30
    for j in range(10):
        ctr = -2.5 + 5.0 * j / 9.0
        s = 0.0
        for i in range(n):
            u = (ctr - z[i]) * inv
            s += math.exp(-0.5 * u * u)
        e[j] = 4.0 * s
        if e[j] > maxl:
            maxl = e[j]
    den = 0.0
    for j in range(10):
        e[j] = math.exp(e[j] - maxl); den += e[j]
    m10 = 0.0
    for j in range(10):
        ctr = -2.5 + 5.0 * j / 9.0
        m10 += (e[j] / den) * ctr
    zs = np.sort(z)
    zmed = zs[(n - 1) // 2]                    # torch lower median
    cmass = 0.0
    for i in range(n):
        if abs(z[i]) < 0.5:
            cmass += 1.0
    out[o] = m10; o += 1
    out[o] = m10 - zmed; o += 1
    out[o] = cmass / n; o += 1

    # ===== family 17: ricker_wavelet (4) =====
    for ri in range(4):
        oc = _conv_same(w, Rk[ri], 1)
        p = 0.0
        for t in range(n):
            if oc[t] > 0.0:
                p += 1.0
        out[o] = p / n; o += 1

    # ===== family 18: hydra_compete (32) — competing-kernel soft win-counts, sqrt-compressed =====
    xd = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        xd[i] = w[i + 1] - w[i]
    for chan in range(2):
        for d in (2, 4):
            for g in range(HYW.shape[0]):
                if chan == 0:
                    r0 = _conv_same(w, HYW[g, 0], d); r1 = _conv_same(w, HYW[g, 1], d)
                    r2 = _conv_same(w, HYW[g, 2], d); r3 = _conv_same(w, HYW[g, 3], d)
                else:
                    r0 = _conv_same(xd, HYW[g, 0], d); r1 = _conv_same(xd, HYW[g, 1], d)
                    r2 = _conv_same(xd, HYW[g, 2], d); r3 = _conv_same(xd, HYW[g, 3], d)
                s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0
                for t in range(r0.shape[0]):
                    v0 = r0[t]; v1 = r1[t]; v2 = r2[t]; v3 = r3[t]
                    am = 0; vm = v0
                    if v1 > vm: am = 1; vm = v1
                    if v2 > vm: am = 2; vm = v2
                    if v3 > vm: am = 3; vm = v3
                    if vm > 0.0:
                        if am == 0: s0 += vm
                        elif am == 1: s1 += vm
                        elif am == 2: s2 += vm
                        else: s3 += vm
                tot = s0 + s1 + s2 + s3 + 1e-8
                out[o] = math.sqrt(s0 / tot); o += 1
                out[o] = math.sqrt(s1 / tot); o += 1
                out[o] = math.sqrt(s2 / tot); o += 1
                out[o] = math.sqrt(s3 / tot); o += 1

    return out



@njit(cache=True)
def _ar_fit(c, p, out, off):
    """AR(p) ridge fit on centered c; writes [log resid_var, log |beta|^2]."""
    n = c.shape[0]
    m = n - p                                  # rows
    A = np.zeros((p, p), np.float64)
    b = np.zeros(p, np.float64)
    for r in range(m):
        t = r + p                              # target index c[t]; predictors c[t-1..t-p]
        for a in range(p):
            xa = c[t - 1 - a]
            b[a] += xa * c[t]
            for d in range(p):
                A[a, d] += xa * c[t - 1 - d]
    for a in range(p):
        A[a, a] += 1e-3
    beta = np.linalg.solve(A, b)
    # residual variance (ddof=1) and coeff norm
    rm = 0.0
    resid = np.empty(m, np.float64)
    for r in range(m):
        t = r + p
        pred = 0.0
        for a in range(p):
            pred += beta[a] * c[t - 1 - a]
        resid[r] = c[t] - pred; rm += resid[r]
    rm /= m
    rv = 0.0
    for r in range(m):
        rv += (resid[r] - rm) ** 2
    rv /= (m - 1)
    bn = 0.0
    for a in range(p):
        bn += beta[a] * beta[a]
    out[off] = math.log(rv + _EPS)
    out[off + 1] = math.log(bn if bn > _EPS else _EPS)


@njit(parallel=True, cache=True)
def _phi_batch(W, GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW):
    B = W.shape[0]
    out = np.empty((B, 173), np.float64)
    for b in prange(B):
        out[b] = _phi_one(W[b], GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW)
    return out


class WMTransformNB:
    """Frozen windowed multi-space encoder phi via njit kernels: (B, L) -> (B, 173)."""

    FAMILIES = [
        ("stats", 12), ("srf_mlp", 12), ("autocorr", 2), ("spectral", 2), ("turning", 2),
        ("trf_gausswin", 12), ("crf_ppv", 32), ("hilbert_env", 2), ("crf_max", 32),
        ("morphology_updown", 4), ("fftbands", 6), ("perm_entropy", 3), ("curvature", 3),
        ("conv_position", 3), ("ar_residual", 4), ("acf_first_min", 3), ("histogram_mode", 3),
        ("ricker_wavelet", 4), ("hydra_compete", 32),
    ]

    def __init__(self, banks=None):
        self._consts = build_consts(banks)

    def phi(self, w):
        W = np.ascontiguousarray(np.asarray(w, np.float64))
        return _phi_batch(W, *self._consts)

    __call__ = phi

    @property
    def n_cols(self):
        return sum(c for _, c in self.FAMILIES)
