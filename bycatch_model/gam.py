"""Penalised-spline GAM (P-splines) in plain numpy.

    eta = b0 + sum(linear terms) + sum_j f_j(x_j),  f_j = cubic B-spline with a
    second-difference penalty (Eilers & Marx). One smoothing parameter, chosen by AIC.

Families: "binomial" (presence/absence, logit link) and "gaussian" (log catch).
"""
from __future__ import annotations

import numpy as np

DEFAULT_LAMBDAS = tuple(np.logspace(-2, 3, 6))


def pspline_basis(x, lo, hi, n_seg=5, degree=3):
    """Equally spaced cubic B-splines on [lo, hi]; x is clamped (flat extrapolation)."""
    x = np.clip(np.asarray(x, float), lo, hi - 1e-9 * (hi - lo))
    h = (hi - lo) / n_seg
    knots = lo + h * np.arange(-degree, n_seg + degree + 1)
    B = ((x[:, None] >= knots[None, :-1]) & (x[:, None] < knots[None, 1:])).astype(float)
    for d in range(1, degree + 1):
        left = (x[:, None] - knots[None, :-(d + 1)]) / (d * h)
        right = (knots[None, d + 1:] - x[:, None]) / (d * h)
        B = left * B[:, :-1] + right * B[:, 1:]
    return B


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class GAM:
    def __init__(self, smooth_terms, linear_terms=(), family="binomial", n_seg=5,
                 lambdas=DEFAULT_LAMBDAS, ridge=1e-4):
        self.smooth_terms = list(smooth_terms)
        self.linear_terms = list(linear_terms)
        self.family = family
        self.n_seg = n_seg
        self.lambdas = tuple(lambdas)
        self.ridge = ridge
        self.ranges = {}
        self.beta = None

    # ---------- design ----------

    def _design(self, F):
        n = len(next(iter(F.values())))
        cols = [np.ones((n, 1))]
        cols += [np.asarray(F[t], float)[:, None] for t in self.linear_terms]
        for t in self.smooth_terms:
            lo, hi = self.ranges[t]
            cols.append(pspline_basis(F[t], lo, hi, self.n_seg))
        return np.hstack(cols)

    def _penalty(self):
        k = self.n_seg + 3
        d2 = np.diff(np.eye(k), n=2, axis=0)
        blocks = [np.zeros((1 + len(self.linear_terms),) * 2)] + [d2.T @ d2] * len(self.smooth_terms)
        size = sum(b.shape[0] for b in blocks)
        S = np.zeros((size, size))
        i = 0
        for b in blocks:
            S[i:i + b.shape[0], i:i + b.shape[0]] = b
            i += b.shape[0]
        return S

    # ---------- fitting ----------

    def fit(self, F, y):
        y = np.asarray(y, float)
        self.ranges = {}
        kept = []
        for t in self.smooth_terms:
            x = np.asarray(F[t], float)
            lo, hi = np.percentile(x, [0.5, 99.5])
            if hi - lo < 1e-9:
                lo, hi = x.min(), x.max()
            if hi - lo < 1e-9:
                continue                    # constant in the training data: drop
            self.ranges[t] = (float(lo), float(hi))
            kept.append(t)
        self.smooth_terms = kept
        X = self._design(F)
        S = self._penalty()
        R = self.ridge * np.eye(X.shape[1])
        best = None
        for lam in self.lambdas:
            beta, edf, dev = self._fit_one(X, y, lam * S + R)
            n = y.size
            crit = dev + 2 * edf if self.family == "binomial" else n * np.log(dev / n + 1e-12) + 2 * edf
            if best is None or crit < best[0]:
                best = (crit, lam, beta, edf, dev)
        _, self.lam, self.beta, self.edf, dev = best
        self.sigma2 = dev / max(y.size - self.edf, 1.0) if self.family == "gaussian" else None
        return self

    def _fit_one(self, X, y, P):
        if self.family == "gaussian":
            XtX = X.T @ X
            A = XtX + P
            beta = np.linalg.solve(A, X.T @ y)
            edf = float(np.trace(np.linalg.solve(A, XtX)))
            return beta, edf, float(np.sum((y - X @ beta) ** 2))
        def penalised_dev(b):
            mu = np.clip(_sigmoid(X @ b), 1e-12, 1 - 1e-12)
            return float(-2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)) + b @ P @ b)

        beta = np.zeros(X.shape[1])
        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        beta[0] = np.log(p0 / (1 - p0))
        pdev = penalised_dev(beta)
        converged = False
        for _ in range(100):
            eta = X @ beta
            mu = np.clip(_sigmoid(eta), 1e-6, 1 - 1e-6)
            w = mu * (1 - mu)
            XtWX = X.T @ (w[:, None] * X)
            A = XtWX + P
            new = np.linalg.solve(A, X.T @ (w * (eta + (y - mu) / w)))
            # Step halving: never accept a step that worsens the penalised deviance
            # (protects against divergence under near-complete separation).
            step = new - beta
            for _ in range(30):
                new_pdev = penalised_dev(beta + step)
                if np.isfinite(new_pdev) and new_pdev <= pdev + 1e-9:
                    break
                step *= 0.5
            beta = beta + step
            change = abs(pdev - new_pdev)
            pdev = new_pdev
            if change < 1e-8 * (1 + abs(pdev)):
                converged = True
                break
        mu = np.clip(_sigmoid(X @ beta), 1e-9, 1 - 1e-9)
        w = mu * (1 - mu)
        XtWX = X.T @ (w[:, None] * X)
        dev = float(-2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        edf = float(np.trace(np.linalg.solve(XtWX + P, XtWX)))
        if not converged or np.max(np.abs(X @ beta)) > 25:
            dev = np.inf                   # unstable fit: never chosen by AIC
        return beta, edf, dev

    # ---------- prediction ----------

    def predict_link(self, F):
        return self._design(F) @ self.beta

    def predict(self, F):
        eta = self.predict_link(F)
        return _sigmoid(eta) if self.family == "binomial" else eta

    def partial(self, term, x):
        """Centred contribution of one smooth term on the link scale (response curve)."""
        i = 1 + len(self.linear_terms)
        for t in self.smooth_terms:
            if t == term:
                lo, hi = self.ranges[t]
                f = pspline_basis(x, lo, hi, self.n_seg) @ self.beta[i:i + self.n_seg + 3]
                return f - f.mean()
            i += self.n_seg + 3
        raise KeyError(term)

    def to_dict(self):
        return {"family": self.family, "smooth_terms": self.smooth_terms, "linear_terms": self.linear_terms,
                "n_seg": self.n_seg, "ranges": self.ranges, "beta": self.beta.tolist(),
                "lambda": self.lam, "edf": self.edf, "sigma2": self.sigma2}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["smooth_terms"], d["linear_terms"], d["family"], d["n_seg"])
        m.ranges = {k: tuple(v) for k, v in d["ranges"].items()}
        m.beta = np.array(d["beta"])
        m.lam, m.edf, m.sigma2 = d["lambda"], d["edf"], d["sigma2"]
        return m
