# src/analogues.py
"""
Analogue-based attribution methods.

All functions return a PN value computed over a neighbourhood of past
atmospheric states that resemble the target circulation.

Note on SLP input
-----------------
These methods expect a 2-D array (T, n_features). Typical options:
  - Global SLP flattened:  slp_2d = data['f_slp'].reshape(T, -1)
  - Local box:             slp_2d = extract_local_slp(data['f_slp'], ...)
  - PCA scores:            slp_2d = PCA(n).fit_transform(slp_flat)

Available methods
-----------------
    run_analogues          : Standard KNN on SLP features (+ optional GMT detrending).
    run_analogues_lasso    : KNN on Lasso-selected SLP dimensions.
    run_analogues_causal   : KNN on Mutual-Information selected SLP dimensions.
    run_analogues_ridge    : Local Ridge-corrected analogue matching.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression

from attribution import compute_pn


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

def get_nonlinear_mb(X, y, threshold=0.01):
    """
    Select SLP features relevant to temperature via Mutual Information
    (nonlinear Markov Blanket approximation).

    Parameters
    ----------
    X         : array (T, n_features)
    y         : array (T,)   Temperature target.
    threshold : float        MI score below which a feature is discarded.

    Returns
    -------
    ndarray of int   Selected indices sorted by decreasing MI score.
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    selected  = np.where(mi_scores > threshold)[0]
    return selected[np.argsort(mi_scores[selected])[::-1]]


# ---------------------------------------------------------------------------
# Analogue methods
# ---------------------------------------------------------------------------

def run_analogues(tas_f, gmt_f, gmt_c, slp_feats, obs_val, t_obs, t_range, mth,
                  n_years=100, n_analogues=100, detrend=True,
                  metric='minkowski', algorithm='auto'):
    """
    Standard KNN analogue method.

    Searches the first n_years of the time series for the n_analogues
    closest past atmospheric states and uses their temperatures as the
    counterfactual distribution.

    Parameters
    ----------
    tas_f      : array (T,)
    gmt_f      : array (T,)         Factual smoothed GMT anomaly.
    gmt_c      : array (T,)         Counterfactual smoothed GMT.
    slp_feats  : array (T, n_feats) SLP representation (PCs, raw box, …).
    obs_val    : float
    t_obs      : int                Time index of the target event.
    t_range    : tuple (t0, t1)
    mth        : str
    n_years    : int                Past-period analogue pool size in years.
    n_analogues: int                Number of nearest neighbours.
    detrend    : bool               Remove GMT trend before matching.
    metric     : str                Distance metric for KNN.
    algorithm  : str                KNN algorithm.

    Returns
    -------
    float  PN value.
    """
    if detrend:
        reg   = LinearRegression().fit(gmt_f[:, None], tas_f)
        tas_f = tas_f - reg.predict(gmt_f[:, None]) + reg.predict(gmt_c[:, None])

    slp_past   = slp_feats[:n_years * 12]
    tas_past   = tas_f[:n_years * 12]
    target     = slp_feats[t_obs].reshape(1, -1)

    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)
    knn.fit(slp_past)
    _, idx = knn.kneighbors(target)
    tas_dyn = tas_past[idx[0]]

    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn, obs_val, method=mth)


def run_analogues_lasso(tas_f, gmt_f, gmt_c, slp_feats, obs_val, t_obs, t_range, mth,
                        n_years=100, n_analogues=100,
                        metric='minkowski', algorithm='auto'):
    """
    KNN analogues with Lasso-based feature selection.

    LassoCV identifies SLP features most predictive of temperature;
    the KNN search is restricted to those dimensions.

    Parameters (same as run_analogues, minus 'detrend')

    Returns
    -------
    float  PN value.
    """
    slp_past = slp_feats[:n_years * 12]
    tas_past = tas_f[:n_years * 12]

    lasso    = LassoCV(cv=5).fit(slp_feats.astype(np.float64), tas_f.astype(np.float64))
    selected = np.where(lasso.coef_ != 0)[0]
    if len(selected) == 0:
        print("Warning: Lasso selected 0 features — falling back to all.")
        selected = np.arange(slp_feats.shape[1])

    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)
    knn.fit(slp_past[:, selected])
    _, idx  = knn.kneighbors(slp_feats[t_obs, selected].reshape(1, -1))
    tas_dyn = tas_past[idx[0]]

    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn, obs_val, method=mth)


def run_analogues_causal(tas_f, gmt_f, gmt_c, slp_feats, obs_val, t_obs, t_range, mth,
                         n_years=100, n_analogues=100, alpha=0.01,
                         metric='minkowski', algorithm='auto'):
    """
    KNN analogues with Mutual-Information-based feature selection.

    A nonlinear Markov Blanket approximation selects SLP features
    causally relevant to temperature.

    Parameters
    ----------
    alpha : float   MI threshold for feature selection (higher = stricter).
    (others same as run_analogues)

    Returns
    -------
    float  PN value.
    """
    slp_past = slp_feats[:n_years * 12]
    tas_past = tas_f[:n_years * 12]

    selected = get_nonlinear_mb(slp_feats.astype(np.float64),
                                tas_f.astype(np.float64), threshold=alpha)
    if len(selected) == 0:
        print("Warning: Markov Blanket empty — falling back to all features.")
        selected = np.arange(slp_feats.shape[1])

    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)
    knn.fit(slp_past[:, selected])
    _, idx  = knn.kneighbors(slp_feats[t_obs, selected].reshape(1, -1))
    tas_dyn = tas_past[idx[0]]

    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn, obs_val, method=mth)


def run_analogues_ridge(tas_f, gmt_f, gmt_c, slp_feats, obs_val, t_obs, t_range, mth,
                        n_years=100, n_analogues=200, detrend=False):
    """
    Local Ridge-corrected analogue method.

    Retrieves a large KNN pool, fits a local Ridge regression
    (temperature ~ SLP) within that pool, then bias-corrects each
    analogue toward the target circulation's predicted temperature.

    Parameters
    ----------
    n_analogues : int   Pool size (recommend >= 200 for stable fit).
    detrend     : bool  Remove GMT trend before matching.
    (others same as run_analogues)

    Returns
    -------
    float  PN value.
    """
    if detrend:
        reg   = LinearRegression().fit(gmt_f[:, None], tas_f)
        tas_f = tas_f - reg.predict(gmt_f[:, None]) + reg.predict(gmt_c[:, None])

    slp_past  = slp_feats[:n_years * 12]
    tas_past  = tas_f[:n_years * 12]
    target    = slp_feats[t_obs].reshape(1, -1)

    knn = NearestNeighbors(n_neighbors=n_analogues, metric='euclidean')
    knn.fit(slp_past)
    _, idx = knn.kneighbors(target)

    X_pool = slp_past[idx[0]]
    y_pool = tas_past[idx[0]]

    ridge       = RidgeCV().fit(X_pool, y_pool)
    pred_target = ridge.predict(target)
    pred_pool   = ridge.predict(X_pool)
    tas_dyn     = y_pool + (pred_target - pred_pool)

    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn, obs_val, method=mth)
