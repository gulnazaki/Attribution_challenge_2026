# src/attribution.py
"""
Core PN computation and built-in attribution methods.

All dynamical adjustment methods receive the global 3-D SLP array
(T, n_lat, n_lon) and call `extract_local_slp` from data_utils to cut
their chosen sub-domain at runtime.

Public API
----------
    compute_pn               : PN kernel shared by all methods.
    run_thermo_ml            : GMT-based thermodynamic adjustment.
    run_dyn_adj_global_pca   : Ridge on global SLP compressed with PCA.
    run_dyn_adj_local        : Ridge on a raw local SLP box (no PCA).
    run_dyn_adj_local_pca    : Ridge on a local SLP box + PCA.
"""

import numpy as np
from scipy.stats import norm, genextreme
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.decomposition import PCA

from data_utils import extract_local_slp


# ---------------------------------------------------------------------------
# PN kernel
# ---------------------------------------------------------------------------

def compute_pn(factual_series, counterfactual_series, threshold, method='empirical'):
    """
    Compute the Probability of Necessity (PN) of an extreme event.

    PN = max(0, 1 - P(X > u | counterfactual) / P(X > u | factual))

    Parameters
    ----------
    factual_series        : array-like (n,)
    counterfactual_series : array-like (m,)
    threshold : float   Observed event value used as exceedance threshold u.
    method    : str     'empirical', 'gaussian', or 'gev'.

    Returns
    -------
    float   PN in [0, 1].  Returns 1e-20 when factual exceedance prob is 0.
    """
    if method == 'empirical':
        p_f  = np.mean(factual_series > threshold)
        p_cf = np.mean(counterfactual_series > threshold)

    elif method == 'gaussian':
        mu_f,  std_f  = norm.fit(factual_series)
        mu_cf, std_cf = norm.fit(counterfactual_series)
        p_f  = norm.sf(threshold, loc=mu_f,  scale=std_f)
        p_cf = norm.sf(threshold, loc=mu_cf, scale=std_cf)

    elif method == 'gev':
        shape_f,  loc_f,  scale_f  = genextreme.fit(factual_series)
        shape_cf, loc_cf, scale_cf = genextreme.fit(counterfactual_series)
        p_f  = genextreme.sf(threshold, shape_f,  loc=loc_f,  scale=scale_f)
        p_cf = genextreme.sf(threshold, shape_cf, loc=loc_cf, scale=scale_cf)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: 'empirical', 'gaussian', 'gev'.")

    if p_f <= 0:
        return 1e-20
    return max(0.0, 1.0 - (p_cf / p_f))


# ---------------------------------------------------------------------------
# Thermodynamic attribution
# ---------------------------------------------------------------------------

def run_thermo_ml(tas_f, gmt_f, gmt_c, obs_val, t_range, mth):
    """
    Thermodynamic attribution via GMT-based linear regression.

    Constructs a counterfactual by removing the GMT-driven trend and
    replacing it with the pre-industrial GMT signal.

    Parameters
    ----------
    tas_f   : array (T,)      Factual temperature time series.
    gmt_f   : array (T,)      Smoothed factual GMT anomaly.
    gmt_c   : array (T,)      Smoothed counterfactual GMT (typically near-flat).
    obs_val : float           Observed event threshold.
    t_range : tuple (t0, t1)  Window for PN distribution.
    mth     : str             PN method.

    Returns
    -------
    float  PN value.
    """
    reg    = LinearRegression().fit(gmt_f[:, None], tas_f)
    tas_cf = tas_f - reg.predict(gmt_f[:, None]) + reg.predict(gmt_c[:, None])
    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_cf[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Dynamical adjustment — global SLP + PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_global_pca(tas_f, slp_f, obs_val, t_range, mth,
                            n_components=50, alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on global SLP compressed with PCA.

    Parameters
    ----------
    tas_f        : array (T,)
    slp_f        : array (T, n_lat, n_lon)  — data['f_slp']
    obs_val      : float
    t_range      : tuple (t0, t1)
    mth          : str
    n_components : int        PCA components to retain (default 50).
    alphas       : array-like RidgeCV regularisation grid.

    Returns
    -------
    float  PN value.
    """
    T        = slp_f.shape[0]
    slp_flat = np.nan_to_num(slp_f.reshape(T, -1))
    pca      = PCA(n_components=n_components)
    slp_pcs  = pca.fit_transform(slp_flat)
    reg      = RidgeCV(alphas=alphas).fit(slp_pcs, tas_f)
    tas_dyn  = reg.predict(slp_pcs)
    t0, t1   = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Dynamical adjustment — local SLP box, no PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_local(tas_f, slp_f, slp_lat, slp_lon, ev_lat, ev_lon,
                      obs_val, t_range, mth,
                      half_width_deg=12.5, alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on a raw local SLP box (no PCA).

    Best suited for small boxes (<= 25x25 deg, ~100 pts on a 2.5-deg grid)
    where Ridge can be applied directly without dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float                    — data['location'][e_idx, 0]
    ev_lon         : float                    — data['location'][e_idx, 1]
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 12.5 -> 25x25 total).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp_f, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    reg       = RidgeCV(alphas=alphas).fit(slp_local, tas_f)
    tas_dyn   = reg.predict(slp_local)
    t0, t1    = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Dynamical adjustment — local SLP box, window only, no PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_local_window(tas_f, slp_f, slp_lat, slp_lon, ev_lat, ev_lon,
                              obs_val, t_range, mth,
                              half_width_deg=25.0, alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on a raw local SLP box, no PCA.

    Ridge is fitted on the full time series (1850-2014) so the regression
    captures the long-term SLP-temperature relationship and produces smooth,
    stable coefficients. PN is then computed on the window [t0, t1].

    Parameters
    ----------
    tas_f          : array (T,)               Full factual temperature series.
    slp_f          : array (T, n_lat, n_lon)  Full global SLP — data['f_slp']
    slp_lat        : array (n_lat,)           data['slp_lat']
    slp_lon        : array (n_lon,)           data['slp_lon']
    ev_lat         : float
    ev_lon         : float
    obs_val        : float
    t_range        : tuple (t0, t1)           Window used only for PN computation.
    mth            : str
    half_width_deg : float   Half-width in degrees (default 25.0 -> 50x50 total).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp_f, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    reg       = RidgeCV(alphas=alphas).fit(slp_local, tas_f)   # full series fit
    tas_dyn   = reg.predict(slp_local)
    t0, t1    = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Dynamical adjustment — local SLP box + PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_local_pca(tas_f, slp_f, slp_lat, slp_lon, ev_lat, ev_lon,
                           obs_val, t_range, mth,
                           half_width_deg=25.0, n_components=20,
                           alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on a local SLP box compressed with PCA.

    Recommended for larger boxes (50x50 deg) where the raw grid is too
    high-dimensional for Ridge without prior dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float
    ev_lon         : float
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 25.0 -> 50x50 total).
    n_components   : int     PCA components (default 20).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp_f, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    pca       = PCA(n_components=min(n_components, slp_local.shape[1]))
    slp_pcs   = pca.fit_transform(np.nan_to_num(slp_local))
    reg       = RidgeCV(alphas=alphas).fit(slp_pcs, tas_f)
    tas_dyn   = reg.predict(slp_pcs)
    t0, t1    = t_range
    return compute_pn(tas_f[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Convenience wrappers used by exploration / visualisation code
# ---------------------------------------------------------------------------

def thermo_cf(tas, gmt):
    """
    Build a thermodynamic counterfactual temperature series.

    Removes the GMT-driven trend and replaces it with the pre-industrial
    level (first value of the factual GMT).

    Parameters
    ----------
    tas : array (T,)   Factual temperature time series.
    gmt : array (T,)   Smoothed factual GMT anomaly.

    Returns
    -------
    ndarray (T,)  Counterfactual temperature.
    """
    gmt_c = np.full_like(gmt, gmt[0])
    reg   = LinearRegression().fit(gmt[:, None], tas)
    return tas - reg.predict(gmt[:, None]) + reg.predict(gmt_c[:, None])


def pn_gaussian(tas_win, cf_win, threshold):
    """
    Quick PN estimate using Gaussian tail probabilities.

    Parameters
    ----------
    tas_win   : array (n,)  Factual temperature sample.
    cf_win    : array (m,)  Counterfactual temperature sample.
    threshold : float       Event exceedance threshold.

    Returns
    -------
    float  PN in [0, 1].
    """
    mu_f,  s_f  = norm.fit(tas_win)
    mu_cf, s_cf = norm.fit(cf_win)
    p_f  = norm.sf(threshold, mu_f,  s_f)
    p_cf = norm.sf(threshold, mu_cf, s_cf)
    return max(0.0, 1.0 - p_cf / p_f) if p_f > 0 else 0.0
