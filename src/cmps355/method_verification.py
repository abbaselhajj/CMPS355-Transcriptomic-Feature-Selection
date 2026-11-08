"""Stage 2 reproducible *synthetic* reference checks for logistic sparse selection.

LASSO (1996) and Elastic Net (2005) originally develop regression penalties;
this study uses explicitly named BINARY LOGISTIC extensions. The basic
half-sample Stability Selection follows Meinshausen--Buehlmann (2010), not the
optional randomized-weight Lasso variant and not StableMate.

No real patient data and no outer holdout cohorts are used in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SparseLogisticSpec:
    """C: inverse regularization; l1_ratio: L1 share (1=LASSO, 0=Ridge)."""

    C: float
    l1_ratio: float
    seed: int = 42
    max_iter: int = 10_000
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be strictly positive and finite")
        if not np.isfinite(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")
        if self.max_iter < 100 or self.tol <= 0:
            raise ValueError("max_iter must be >=100; tol must be positive")


def sparse_logistic_model(spec: SparseLogisticSpec) -> Pipeline:
    """Fit scaling inside the pipeline: the scaler MUST see training rows only.

    scikit-learn's SAGA solver optimizes mean binary log loss plus
    [alpha * ||w||_1 + (1-alpha) * ||w||_2^2 / 2] / (n * C).
    The intercept is not regularized. This is the logistic extension of the
    published penalties, not the original paper's exact regression example.
    scikit-learn >=1.8 uses l1_ratio in place of the deprecated penalty flag.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            solver="saga", l1_ratio=spec.l1_ratio, C=spec.C,
            fit_intercept=True, random_state=spec.seed,
            max_iter=spec.max_iter, tol=spec.tol,
        )),
    ])


def _validate_data(X: NDArray, y: NDArray, *, unique_features: bool = False) -> tuple[NDArray, NDArray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] < 2:
        raise ValueError("X must be a 2D numeric matrix, >=8 rows and >=2 columns")
    if y.ndim != 1 or len(y) != len(X) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("y must be a 0/1 vector with both classes and one label per row")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite numbers")
    return X, y.astype(np.int64, copy=False)


def penalty_objective(X: NDArray, y: NDArray, weights: NDArray,
                      intercept: float, spec: SparseLogisticSpec) -> float:
    """Independent evaluation of the exact fitted regularized loss (scaled X)."""
    X, y = _validate_data(X, y)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (X.shape[1],):
        raise ValueError("weights dimension mismatch")
    z = X @ weights + float(intercept)
    mean_logloss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    n = len(y)
    regularization = (
        spec.l1_ratio * np.abs(weights).sum()
        + (1 - spec.l1_ratio) * weights.dot(weights) / 2
    ) / (n * spec.C)
    return float(mean_logloss + regularization)


def independent_reference_solver(X_scaled: NDArray, y: NDArray,
                                 spec: SparseLogisticSpec) -> tuple[NDArray, float, float]:
    """Independent convex solution via scipy L-BFGS-B split variables u,v >= 0.

    w=u-v makes the L1 penalty differentiable in constrained coordinates. This
    numerical reference does NOT call scikit-learn or SAGA. Intended ONLY for
    tiny synthetic reference problems; not a production high-dimensional fit.
    """
    X, y = _validate_data(X_scaled, y)
    n, p = X.shape
    alpha, lam = spec.l1_ratio, 1.0 / (n * spec.C)

    def fun(theta: NDArray) -> tuple[float, NDArray]:
        u, v, b = theta[:p], theta[p:2*p], theta[-1]
        w = u - v
        z = X @ w + b
        loss = float(np.mean(np.logaddexp(0, z) - y * z))
        value = loss + lam * (alpha * (u.sum() + v.sum()) + 0.5 * (1-alpha) * w.dot(w))
        residual = expit(z) - y
        grad_w = (X.T @ residual) / n + lam * (1-alpha) * w
        grad_u = grad_w + lam * alpha
        grad_v = -grad_w + lam * alpha
        grad_b = float(np.mean(residual))
        return float(value), np.r_[grad_u, grad_v, grad_b]

    start = np.zeros(2*p+1)
    bounds = [(0, None)] * (2*p) + [(None, None)]
    result = minimize(fun, start, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": 20_000, "ftol": 1e-13, "gtol": 1e-8})
    if not result.success and np.linalg.norm(result.jac, np.inf) > 1e-4:
        raise RuntimeError(f"Independent solver did not converge: {result.message}")
    weights = result.x[:p] - result.x[p:2*p]
    return weights, float(result.x[-1]), float(result.fun)


def synthetic_binary_data(seed: int = 19) -> tuple[NDArray, NDArray, tuple[str, ...]]:
    """Fixed artificial problem with no biological interpretation."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(160, 12))
    logits = 2.8 * X[:, 0] - 2.3 * X[:, 1] + 1.5 * X[:, 2]
    y = rng.binomial(1, expit(logits))
    return X, y, tuple(f"synthetic_{i:02d}" for i in range(X.shape[1]))


@dataclass
class StabilityResult:
    """Per-C estimated selection probabilities, then MAX across tested C values.

    The stable set is NOT necessarily predictive; inferences are conditional
    on the subsample specification and the predeclared C grid. Thresholding
    does not justify a theoretical PFER claim without theorem assumptions.
    """
    feature_names: list[str]
    C_grid: list[float]
    n_subsamples: int
    half_sample_size: int
    threshold: float
    coef_cutoff: float
    seed: int
    per_C_frequencies: list[list[float]]
    max_frequencies: list[float]
    stable_features: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def stability_selection(X: NDArray, y: NDArray, feature_names: Sequence[str],
                        C_grid: Sequence[float], *, n_subsamples: int = 30,
                        threshold: float = 0.8, coef_cutoff: float = 1e-5,
                        seed: int = 42) -> StabilityResult:
    """Basic M-B 2010 stability selection: random WITHOUT-replacement half-samples.

    For each half-sample and each C in a *fixed* grid: fit a L1-logistic
    selector, record whether each coefficient is nonzero (above numerical
    cutoff), estimate P(select | C) by B-frequency, and retain genes whose
    max_C P(select | C) >= threshold. This is NOT choosing a separate C by
    optimizing the labels of the held-out test cohort. It does not implement
    the paper's optional randomized-weights extension.

    Each fit's StandardScaler is trained on just that half-sample. The method
    is computationally expensive on 10k genes; use only synthetic tests here.
    """
    X, y = _validate_data(X, y)
    features = list(feature_names)
    grid = [float(c) for c in C_grid]
    if len(features) != X.shape[1] or len(set(features)) != len(features):
        raise ValueError("feature_names must uniquely identify every X column")
    if not grid or len(set(grid)) != len(grid) or any((not np.isfinite(c) or c <= 0) for c in grid):
        raise ValueError("C_grid must contain distinct, strictly positive finite values")
    if not isinstance(n_subsamples, int) or n_subsamples < 2:
        raise ValueError("At least two subsamples are required")
    if not (0.5 < threshold <= 1):
        raise ValueError("The stability threshold must be in (0.5, 1]")
    if coef_cutoff <= 0 or not np.isfinite(coef_cutoff):
        raise ValueError("coef_cutoff must be positive and finite")
    rng = np.random.default_rng(seed)
    n_half = len(y) // 2
    counts = np.zeros((len(grid), X.shape[1]), dtype=np.int64)
    for b in range(n_subsamples):
        idx = rng.choice(len(y), size=n_half, replace=False)
        if len(np.unique(y[idx])) != 2:
            raise ValueError("A half-sample has a single class: unable to estimate logistic coefficients")
        for i, C in enumerate(grid):
            pipe = sparse_logistic_model(SparseLogisticSpec(C=C, l1_ratio=1.0, seed=seed + b))
            pipe.fit(X[idx], y[idx])
            selected = np.abs(pipe.named_steps["logistic"].coef_.ravel()) > coef_cutoff
            counts[i] += selected.astype(np.int64)
    freqs = counts / n_subsamples
    peak = freqs.max(axis=0)
    retained = [name for name, prob in zip(features, peak) if prob >= threshold]
    return StabilityResult(features, grid, n_subsamples, n_half,
                           threshold, coef_cutoff, seed,
                           freqs.tolist(), peak.tolist(), retained)
