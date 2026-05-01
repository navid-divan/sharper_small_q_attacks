"""
linfty_smallqSIS.py
-------------------
ell_infty Z-shape attack on ISIS_n,m,q,beta_infty
(the Dilithium-flavoured analogue of [DEP23] / efficient_smallqSIS.py).

The geometry of the attack is identical to the ell_2 case:
    1.  Build kernel basis B_A of Lambda_q^perp(A).
    2.  Run BKZ-beta -> Z-shape profile, n_q q-vectors in Zone I.
    3.  Sieve a rank-beta_S projected sublattice past Zone I, obtaining
        a database of short ell_2 vectors.
    4.  Lift each candidate over the n_q q-vectors by reducing mod q
        around zero, asking whether the lifted vector lies in the
        ell_infty box of half-side beta_infty.

The KEY simplification in the ell_infty case is the closed-form per-lift
probability for uniform-mod-q residues:
        p_lift_infty(beta_infty; q, n_q)
            = ((2 beta_infty + 1) / q) ** n_q
no theta-function convolution required. The ISIS reduction sharpens this
to
        p_ISIS_infty_sharp = (2/q) * ((2 beta_infty + 1) / q) ** (n_q - 1)
when the residue at index zero must lie in {+/-1} (always inside the box
for any beta_infty >= 1).

Drops in next to small_qSIS.py / efficient_smallqSIS.py.

Public entry points:
    log_prob_ball_infty(n_q, beta_infty, q)       -- log SIS* lift prob
    log_prob_isis_infty_sharp(n_q, beta_infty, q) -- log ISIS sharp prob
    sieve_box_fraction(beta_S, gh, beta_infty)    -- frac of sieve DB
                                                     surviving the box test
                                                     on Zone II coordinates
    log_expected_successes_infty(...)             -- integrated count
    sharp_cost_infty(q, w, h, beta_infty, beta)   -- one-shot cost
    optimize_sharp_infty(q, w, h, beta_infty)     -- 1D blocksize search

Notes
-----
* For Dilithium-2 (n=1024, q=8380417, beta_infty=523776) the box is so
  large that sieve-tail coordinates trivially satisfy it; the only thing
  that gates success is the Zone I lift event, whose probability is
  ((2 beta_infty + 1)/q)^{n_q} -- a quantity the engine still has to
  trade against the BKZ block-size cost.
* The cost we report is again CoreSVP: kappa * beta - log2(E[succ]),
  with kappa = log2(sqrt(3/2)) ~= 0.292.
"""

from math import log, sqrt, pi, exp, erf, ceil
from modelBKZ import (svp_classical, BKZ_first_length,
                      construct_BKZ_shape)

log_infinity = 9999


# ---------------------------------------------------------------------------
# (1)  Closed-form ell_infty cube-box intersection probabilities
# ---------------------------------------------------------------------------
def log_prob_ball_infty(n_q, beta_infty, q):
    """
    log P[ ||x||_infty <= beta_infty ]   for  x ~ U(Cube_{n_q}(q)).
    Closed form: ((2 beta_infty + 1) / q) ** n_q.
    """
    if n_q < 0:
        return -log_infinity
    if n_q == 0:
        return 0.0
    side = min(2 * beta_infty + 1, q)        # never exceed q
    if side <= 0:
        return -log_infinity
    return n_q * log(side / q)


def log_prob_isis_infty_sharp(n_q, beta_infty, q):
    """
    Sharp ISIS_infty per-lift probability:
        log P[ x_1 in {+/-1}  AND  ||x||_infty <= beta_infty ]
      = log(2/q) + log P[ ||(x_2,...,x_{n_q})||_infty <= beta_infty ]
      = log(2/q) + (n_q - 1) * log((2 beta_infty + 1) / q).

    For any beta_infty >= 1 the constraint x_1 in {+/-1} is INSIDE the
    box, so the ISIS reduction adds the factor 2/q exactly (no
    correction needed -- this is sharp by construction in the
    ell_infty case).
    """
    if n_q < 1 or beta_infty < 1:
        return -log_infinity
    if n_q == 1:
        return log(2.0 / q)
    return log(2.0 / q) + log_prob_ball_infty(n_q - 1, beta_infty, q)


# ---------------------------------------------------------------------------
# (2)  Box-survival fraction of sieve outputs (Zone II/III tail)
# ---------------------------------------------------------------------------
def sieve_box_fraction(beta_S, gh, beta_infty, otf=True,
                       num_buckets=24):
    """
    Estimate the fraction of sieve-output vectors v in Lambda_{[ell:r]}
    whose Zone II / III coordinates already satisfy ||v||_infty <=
    beta_infty.

    Heuristic.  Assume the projected sieve vector v is approximately
    isotropic with squared length ||v||^2 = L^2.  Each coordinate then
    has variance L^2 / beta_S, so heuristically v_i ~ N(0, L^2/beta_S).
    The probability that all beta_S coordinates fall in
    [-beta_infty, beta_infty] is (under independence):

        survive(L) = ( erf( beta_infty / (L sqrt(2/beta_S)) ) ) ** beta_S.

    We integrate against the BDGL spherical-shell density
    f_L(L) propto L^{beta_S - 1} on [0, sqrt(3/2) gh] (or place a Dirac
    at sqrt(4/3) gh if otf=False, mirroring the conservative model).
    Returns the *log* of the surviving expected count, i.e.
        log( |P| * E_L[ survive(L) ] ).
    """
    if not otf:
        L = sqrt(4.0 / 3.0) * gh
        sigma = L / sqrt(beta_S)
        if sigma <= 0:
            log_surv = 0.0
        else:
            arg = beta_infty / (sigma * sqrt(2.0))
            p_one = erf(arg)
            if p_one <= 0:
                log_surv = -log_infinity
            else:
                log_surv = beta_S * log(p_one)
        log_db = (beta_S / 2.0) * log(4.0 / 3.0)
        return log_db + log_surv

    L_max = sqrt(3.0 / 2.0) * gh
    log_db = (beta_S / 2.0) * log(3.0 / 2.0)

    # bucketed BDGL shell density.
    # Integrate over u = L / L_max in [0, 1].
    # frac(bucket) = u_hi^beta_S - u_lo^beta_S, computed without overflow.
    log_total = -log_infinity
    for k in range(num_buckets):
        u_lo = k / num_buckets
        u_hi = (k + 1) / num_buckets
        # u_hi^b - u_lo^b ; both in [0,1] so no overflow even at large b
        try:
            frac = (u_hi ** beta_S) - (u_lo ** beta_S)
        except OverflowError:
            frac = 0.0
        if frac <= 0.0:
            continue
        L_mid = 0.5 * (u_lo + u_hi) * L_max
        sigma = L_mid / sqrt(beta_S) if beta_S > 0 else 1e-300
        if sigma <= 0:
            continue
        arg = beta_infty / (sigma * sqrt(2.0))
        p_one = erf(arg)
        if p_one <= 0:
            continue
        log_surv = beta_S * log(p_one)
        term = log(frac) + log_surv
        if log_total == -log_infinity:
            log_total = term
        else:
            m = max(log_total, term)
            log_total = m + log(exp(log_total - m) + exp(term - m))
    if log_total == -log_infinity:
        return -log_infinity
    return log_db + log_total


# ---------------------------------------------------------------------------
# (3)  Integrated expected successes
# ---------------------------------------------------------------------------
def log_expected_successes_infty(q, n_q, beta_infty, beta_S, gh_sieve,
                                 otf=True, inhom="specific"):
    """
    log E[ # successful lifts ]   under the ell_infty engine.

    inhom:
        None        : SIS_infty                -> log_prob_ball_infty
        "specific"  : sharp ISIS_infty         -> log_prob_isis_infty_sharp
    """
    log_db_surv = sieve_box_fraction(beta_S, gh_sieve, beta_infty,
                                     otf=otf)
    if log_db_surv <= -log_infinity:
        return -log_infinity

    if inhom == "specific":
        log_p_lift = log_prob_isis_infty_sharp(n_q, beta_infty, q)
    else:
        log_p_lift = log_prob_ball_infty(n_q, beta_infty, q)
    if log_p_lift <= -log_infinity:
        return -log_infinity

    return log_db_surv + log_p_lift


# ---------------------------------------------------------------------------
# (4)  Sharp cost  +  1D blocksize optimisation
# ---------------------------------------------------------------------------
def sharp_cost_infty(q, w, h, beta_infty, beta,
                     cost_svp=svp_classical, otf=True, inhom="specific",
                     beta_sieve=None, verbose=False):
    """
    Sharp ell_infty cost at single BKZ blocksize beta.
    By default (beta_sieve=None): beta_R = beta_S = beta (1D model).
    If beta_sieve is supplied: BKZ at `beta`, terminal sieve at
    `beta_sieve`. This matches the practical attack which does
    progressive BKZ to a modest beta then sieves a larger sub-block.
    """
    w_eff = w + 1 if inhom is not None else w     # SIS* rank bump
    (n_q, _, profile) = construct_BKZ_shape(q, h, w_eff - h, beta)

    if n_q == 0:
        first = BKZ_first_length(q, h, w_eff - h, beta)
        # rough infty bound from ell_2: ||x||_infty <= ||x||
        return log_infinity if first > beta_infty else cost_svp(beta)

    bs = beta if beta_sieve is None else beta_sieve
    if n_q + bs > w_eff:
        beta_use = w_eff - n_q
    else:
        beta_use = bs
    if beta_use < 20:
        return log_infinity

    log_vol = sum(profile[n_q : n_q + beta_use])
    gh_sieve = sqrt(beta_use / (2.0 * pi * exp(1))) * exp(log_vol / beta_use)

    log_n = log_expected_successes_infty(q, n_q, beta_infty,
                                         beta_use, gh_sieve,
                                         otf=otf, inhom=inhom)
    if log_n <= -log_infinity:
        return log_infinity

    log_p_succ = min(0.0, log_n)
    # CoreSVP cost uses the dominating dimension
    cost = cost_svp(max(beta, beta_use)) - log_p_succ / log(2.0)
    if verbose:
        print("  beta_BKZ=%d, beta_sieve=%d | n_q=%d | gh_S=%.2f | "
              "log2 E[succ]=%.3f | cost=%.3f"
              % (beta, beta_use, n_q, gh_sieve, log_n / log(2.0), cost))
    return cost


def optimize_sharp_infty(q, w, h, beta_infty,
                         cost_svp=svp_classical,
                         otf=True, inhom="specific",
                         beta_range=None, verbose=False):
    """
    1D minimisation over BKZ block-size beta.
    Returns (best_cost, best_beta).
    """
    if beta_range is None:
        beta_range = range(30, min(300, w) + 1, 2)
    best = (log_infinity, None)
    for b in beta_range:
        c = sharp_cost_infty(q, w, h, beta_infty, b,
                             cost_svp=cost_svp, otf=otf, inhom=inhom)
        if c < best[0]:
            best = (c, b)
    if verbose and best[1] is not None:
        print("  OPTIMAL: beta = %d, cost = %.2f bits"
              % (best[1], best[0]))
    return best


# ---------------------------------------------------------------------------
# (5)  Reference Dilithium parameters (for downstream scripts)
# ---------------------------------------------------------------------------
DILITHIUM = {
    # Round-3 specs.  beta_infty here is gamma_1 - beta (the bound that
    # the forged signature 'z' must satisfy to be accepted).
    "Dilithium2": {"k": 4, "ell": 4, "n": 256, "q": 8380417,
                   "gamma1": 1 << 17, "beta_chal": 78,
                   "beta_infty": (1 << 17) - 78},
    "Dilithium3": {"k": 6, "ell": 5, "n": 256, "q": 8380417,
                   "gamma1": 1 << 19, "beta_chal": 196,
                   "beta_infty": (1 << 19) - 196},
    "Dilithium5": {"k": 8, "ell": 7, "n": 256, "q": 8380417,
                   "gamma1": 1 << 19, "beta_chal": 120,
                   "beta_infty": (1 << 19) - 120},
}


def dilithium_dims(name):
    """Returns (n_total, m_total, q, beta_infty) of the lattice that the
    attack targets: the kernel of an n_total-by-m_total matrix mod q,
    where n_total = k*256, m_total = (k+ell)*256."""
    p = DILITHIUM[name]
    n_total = p["k"] * p["n"]
    m_total = (p["k"] + p["ell"]) * p["n"]
    return n_total, m_total, p["q"], p["beta_infty"]
