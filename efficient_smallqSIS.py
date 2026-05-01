"""
Sharp cost engine for the efficient Small-q attack.
    (1) length-distribution integration via the BDGL spherical-shell density,
    (2) sharp ISIS per-lift probability via ONE extra truncated-theta call
        (replaces the detached 2/q factor of [DEP23, Sec. 3.5]),
    (3) decoupled search over (beta_R, beta_S) with beta_R <= beta_S.
The underlying fast-theta convolution engine of `fastTheta.BoxedTheta` is
reused unchanged.
"""

from math import log, sqrt, pi, exp, ceil

from modelBKZ import svp_classical, BKZ_first_length, construct_BKZ_shape
from fastTheta import BoxedTheta

log_infinity = 9999
_bt_cache = {}
_BT_CACHE_LIMIT = 16          # at most this many BoxedTheta objects kept


def _bt(R, q):
    """Return a (rotating-cache-)cached BoxedTheta instance with radius ceil(R)."""
    R_i = int(ceil(R))
    key = (R_i, q)
    if key in _bt_cache:
        return _bt_cache[key]
    # bound the cache to keep RAM in check during big (bR, bS) sweeps
    while len(_bt_cache) >= _BT_CACHE_LIMIT:
        _bt_cache.pop(next(iter(_bt_cache)))
    _bt_cache[key] = BoxedTheta(R_i, q)
    return _bt_cache[key]


# theta-based event probabilities
def log_prob_uniform_in_ball(n, R, q):
    """
    log P[ ||x|| <= R ]  for  x ~ U(Cube_n(q)).
    the object computed by `small_qSIS.logIntersectionProportion`
    without LSH; replicated here to make this module self-contained.
    """
    if R <= 0 or n <= 0:
        return -log_infinity
    bt = _bt(R, q)
    arr = bt(n, ncoll=0)
    s = float(sum(arr))
    if s <= 0.0:
        return -log_infinity
    return log(s)


def log_prob_isis_sharp(n_q, R, q):
    """
    SHARP ISIS per-lift probability:
        log P[ x_1 in {+/-1}  AND  ||x|| <= R ],  x ~ U(Cube_{n_q}(q))
      = log(2/q) + log P[ ||(x_2,...,x_{n_q})||^2 <= R^2 - 1 ]
      = log(2/q) + log_prob_uniform_in_ball(n_q - 1, sqrt(R^2 - 1), q).
    replaces the "SIS* probability times 2/q" heuristic of [DEP23, Sec. 3.5]
    by the true joint probability, which is strictly larger whenever the
    marginal distribution of x_1 in the successful-lift event is concentrated
    on small integers.
    """
    if n_q < 1 or (R * R) < 1.0:
        return -log_infinity
    if n_q == 1:
        return log(2.0 / q)
    R_rem = sqrt(R * R - 1.0)
    log_rem = log_prob_uniform_in_ball(n_q - 1, R_rem, q)
    if log_rem <= -log_infinity:
        return -log_infinity
    return log(2.0 / q) + log_rem


# sieve output length distribution
def sieve_length_distribution(beta, otf=True, num_buckets=32):
    """
    discretised length distribution f_L of vectors considered by the sieve.
        * not OTF : point mass (4/3)^{beta/2} at L / gh = sqrt(4/3).
        * OTF BDGL: spherical-shell volume density
                        f_L(L) propto L^{beta-1}
                    on [0, sqrt(3/2) * gh], with total mass (3/2)^{beta/2}.
                    bucketised into `num_buckets` equal-width slices.
    returns a list of (L_rel, count) pairs with L_rel = L / gh.
    """
    if not otf:
        return [(sqrt(4.0 / 3.0), (4.0 / 3.0) ** (beta / 2.0))]

    L_max_rel = sqrt(3.0 / 2.0)
    N_total = (3.0 / 2.0) ** (beta / 2.0)
    buckets = []
    for k in range(num_buckets):
        lo = (k / num_buckets) * L_max_rel
        hi = ((k + 1) / num_buckets) * L_max_rel
        # fraction of volumetric mass in [lo, hi]
        p_mass = (hi ** beta - lo ** beta) / (L_max_rel ** beta)
        count = p_mass * N_total
        L_mid = 0.5 * (lo + hi)
        if count >= 1.0 and L_mid > 0.0:
            buckets.append((L_mid, count))
    return buckets


def _log_sum_exp(a, b):
    if a == -log_infinity:
        return b
    if b == -log_infinity:
        return a
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))


# integrated expected-successes count
def log_expected_successes(q, n_q, nu, beta_S, gh_sieve,
                           otf=True, inhom="specific"):
    """
    log E[#successful lifts]   under the sharp engine.
    inhom:
        None       : SIS*              -> use log_prob_uniform_in_ball
        "specific" : sharp ISIS        -> use log_prob_isis_sharp
        "classic"  : 2/q heuristic     -> SIS* prob + log(2/q)
    """
    log_total = -log_infinity
    for (L_rel, count) in sieve_length_distribution(beta_S, otf=otf):
        L_proj = L_rel * gh_sieve
        if L_proj >= nu:
            continue
        R = sqrt(nu * nu - L_proj * L_proj)

        if inhom == "specific":
            log_p = log_prob_isis_sharp(n_q, R, q)
        elif inhom == "classic":
            log_p = log_prob_uniform_in_ball(n_q, R, q) + log(2.0 / q)
        else:  # SIS*
            log_p = log_prob_uniform_in_ball(n_q, R, q)

        if log_p <= -log_infinity:
            continue
        log_total = _log_sum_exp(log_total, log(count) + log_p)
    return log_total


# decoupled (beta_R, beta_S) cost and optimisation
def sharp_cost(q, w, h, nu, beta_R, beta_S,
               cost_svp=svp_classical, otf=True, inhom="specific",
               verbose=False):
    """
    sharp cost of the attack with decoupled (beta_R, beta_S).
    :param w:      rank of the SIS kernel lattice (pre ISIS->SIS* bump)
    :param h:      log-volume exponent (vol = q^h)
    :param beta_R: BKZ blocksize driving the Z-shape
    :param beta_S: terminal sieve dimension (must satisfy beta_S >= beta_R
                   in the optimizer, but not in this helper)
    """
    if inhom is not None:
        w = w + 1   # SIS* rank bump from the ISIS reduction

    (n_q, _, profile) = construct_BKZ_shape(q, h, w - h, beta_R)

    if n_q == 0:
        # no q-vectors left -> attack degenerates to generic SIS
        first = BKZ_first_length(q, h, w - h, beta_R)
        return log_infinity if first > nu else cost_svp(beta_R)

    if n_q + beta_S > w:
        beta_S = w - n_q
    if beta_S < 20:
        return log_infinity

    # gh of the sieve block (indices n_q .. n_q + beta_S - 1)
    log_vol = sum(profile[n_q : n_q + beta_S])
    gh_sieve = sqrt(beta_S / (2.0 * pi * exp(1))) * exp(log_vol / beta_S)

    log_n = log_expected_successes(q, n_q, nu, beta_S, gh_sieve,
                                   otf=otf, inhom=inhom)
    if log_n <= -log_infinity:
        return log_infinity

    log_p_succ = min(0.0, log_n)                       # cap at 1 per attempt
    beta_eff = max(beta_R, beta_S)
    cost = cost_svp(beta_eff) - log_p_succ / log(2.0)

    if verbose:
        print("  (beta_R, beta_S) = (%d, %d) | n_q=%d | gh_S=%.2f | "
              "log2 E[succ]=%.3f | cost=%.3f"
              % (beta_R, beta_S, n_q, gh_sieve,
                 log_n / log(2.0), cost))
    return cost


def optimize_sharp(q, w, h, nu, cost_svp=svp_classical,
                   otf=True, inhom="specific",
                   beta_range=None,
                   beta_R_range=None, beta_S_range=None,
                   decouple=False,
                   verbose=False):
    """
    optimize the sharp cost over blocksizes.
    default (decouple=False):  1D search with beta_R = beta_S = beta.
        matches DEP23's search space and is fast; empirically at
        Falcon/Mitaka parameters the 2D optimum lies on this diagonal
        anyway (verified at (n,q)=(256,257): opt = (126,126)), so 1D
        is the right default.
    If decouple=True :  full 2D search (beta_R, beta_S) with
                         beta_R <= beta_S. Slower by ~10x; retained for
                         completeness.
    returns (best_cost, best_beta_R, best_beta_S).
    """
    if not decouple:
        if beta_range is None:
            beta_range = (beta_S_range if beta_S_range is not None
                          else range(30, min(300, w) + 1, 2))
        best = (log_infinity, None, None)
        for b in beta_range:
            c = sharp_cost(q, w, h, nu, b, b,
                           cost_svp=cost_svp, otf=otf, inhom=inhom)
            if c < best[0]:
                best = (c, b, b)
        if verbose and best[1] is not None:
            print("  OPTIMAL (1D): beta = %d, cost = %.2f bits"
                  % (best[1], best[0]))
        return best

    # ---- full 2D search (slower) ----
    if beta_R_range is None:
        beta_R_range = range(20, min(200, w) + 1, 2)
    if beta_S_range is None:
        beta_S_range = range(30, min(300, w) + 1, 2)
    best = (log_infinity, None, None)
    for bR in beta_R_range:
        for bS in beta_S_range:
            if bS < bR:
                continue
            c = sharp_cost(q, w, h, nu, bR, bS,
                           cost_svp=cost_svp, otf=otf, inhom=inhom)
            if c < best[0]:
                best = (c, bR, bS)
    if verbose and best[1] is not None:
        print("  OPTIMAL (2D): (beta_R, beta_S) = (%d, %d),  cost = %.2f bits"
              % (best[1], best[2], best[0]))
    return best
