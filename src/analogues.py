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


def run_analogues_local(tas_f, gmt_f, gmt_c, slp_f, slp_lat, slp_lon,
                         ev_lat, ev_lon, obs_val, t_obs, t_range, mth,
                         half_width_deg=25.0, n_analogues=100,
                         metric='euclidean', algorithm='auto'):
    """
    Local analogue method: KNN search on a lat/lon box around the event
    barycentre, using the full time series as the analogue pool.

    The counterfactual distribution is the temperatures of the k nearest
    neighbours (direct, no detrending, no bias correction). The full
    record (all available years) is used as the pool so the search is
    not artificially limited to an early pre-warming period.

    Unlike the global analogue methods, the SLP distance is computed only
    within the local box, making it more sensitive to the regional
    circulation pattern that directly drives the event.

    Parameters
    ----------
    tas_f          : array (T,)               Full factual temperature series.
    gmt_f          : array (T,)               Smoothed factual GMT anomaly.
    gmt_c          : array (T,)               Smoothed counterfactual GMT.
    slp_f          : array (T, n_lat, n_lon)  Full global SLP — data['f_slp']
    slp_lat        : array (n_lat,)           data['slp_lat']
    slp_lon        : array (n_lon,)           data['slp_lon']
    ev_lat         : float                    data['location'][e_idx, 0]
    ev_lon         : float                    data['location'][e_idx, 1]
    obs_val        : float                    Event threshold.
    t_obs          : int                      Time index of the event.
    t_range        : tuple (t0, t1)           Window for PN computation.
    mth            : str                      PN estimator.
    half_width_deg : float
        Half-width of the local SLP box in degrees (default 25.0 → 50×50°).
        Should match the spatial scale of the circulation feature driving
        the event; consistent with ANALOGUE_HALF_DEG in the notebooks.
    n_analogues    : int    Number of nearest neighbours (default 100).
    metric         : str    Distance metric for KNN (default 'euclidean').
    algorithm      : str    KNN algorithm.

    Returns
    -------
    float  PN value.
    """
    from data_utils import extract_local_slp

    # Extract the local SLP box — (T, n_pts)
    slp_local = extract_local_slp(
        slp_f, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)

    # KNN on full record, excluding the event timestep itself
    T = slp_local.shape[0]
    pool_idx = np.array([t for t in range(T) if t != t_obs])
    knn = NearestNeighbors(n_neighbors=n_analogues,
                            metric=metric, algorithm=algorithm)
    knn.fit(slp_local[pool_idx])
    _, idx = knn.kneighbors(slp_local[t_obs].reshape(1, -1))

    # Temperatures of the nearest neighbours → counterfactual distribution
    tas_dyn = tas_f[pool_idx[idx[0]]]

    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn, obs_val, method=mth)
