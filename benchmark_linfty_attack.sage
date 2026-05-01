"""
ell_infty Z-shape attack on Dilithium-style ISIS:
    given A in Z_q^{n x m} and u in Z_q^n, find x in Z^m
    with A x = u (mod q) and ||x||_infty <= beta_infty.
re-instantiation of DEP23's attack.sage with the success
criterion swapped from ||y||<nu (ell_2) to ||y||_inf<=beta_infty.
ell_infty Z-shape becomes tractable at SMALL modulus (q <= 31)
with MODERATE BKZ (beta_BKZ = 18): smaller q makes the q-vectors
shorter, so cheap BKZ absorbs most of them, and the closed-form
box probability ((2 beta_infty+1)/q)^{n_q} stays O(1).
the closed-form
    pi^infty(beta_infty; q, n_q) = ((2 beta_infty + 1)/q)^{n_q}
replaces the truncated-theta convolution of [DEP23], making both the
cost estimate and the attack itself strictly simpler than ell_2.
"""

from fpylll import BKZ, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from sage.all import (IntegerModRing, ZZ, matrix, identity_matrix,
                      random_matrix, random_vector, vector, set_random_seed)
from g6k import Siever

from numpy import array, zeros, abs as npabs
from numpy.linalg import norm

import sys
import time


# presets by linfty_smallqSIS (sharp cost model) to have
# log2 E[#succ]/trial >= +3.6 at these (beta_BKZ, beta_sieve) settings.
# BKZ-18 is cheap on these dimensions.
PRESETS = {
    "fast": {
        "n": 56, "q": 23, "beta_infty": 10,
        "z1": 56, "z2": 56,
        "beta_BKZ": 18, "beta_sieve": 30,
        "predicted_log2_succ": 4.23,
    },
    "medium": {
        "n": 64, "q": 31, "beta_infty": 14,
        "z1": 64, "z2": 64,
        "beta_BKZ": 18, "beta_sieve": 30,
        "predicted_log2_succ": 3.76,
    },
    "small": {
        "n": 48, "q": 13, "beta_infty": 5,
        "z1": 48, "z2": 48,
        "beta_BKZ": 18, "beta_sieve": 30,
        "predicted_log2_succ": 3.65,
    },
}


def run_one_trial(n, q, beta_infty, z1, z2, beta_BKZ, beta_sieve):
    """DEP23's attack.sage pipeline with the ell_infty success criterion."""
    m_ = 2 * n
    m = m_ + 1
    Zq = IntegerModRing(q)
    Id = identity_matrix

    A0 = random_matrix(Zq, m_ - n, n)
    AISIS = matrix.block(Zq, [[identity_matrix(ZZ, n)], [A0]])
    u = random_vector(Zq, n)

    Au0 = matrix.block(Zq, [[matrix(u)], [A0]])
    Afull = matrix.block(Zq, [[identity_matrix(ZZ, n)], [Au0]])
    A_np = array(Afull.lift())

    B = matrix.block(ZZ, [[q * Id(n), 0], [-Au0, Id(m - n)]])
    d = z1 + z2
    B_ = matrix(B)[n - z1 : n + z2, n - z1 : n + z2]

    def complete_solution(v):
        x = zeros(m, dtype="int64")
        x[n - z1 : n + z2] = v
        y = (x.dot(A_np))[: n - z1] % q
        y -= q * (y > q / 2)
        x[: n - z1] = -y
        return x

    C = B_.LLL()
    X = IntegerMatrix.from_matrix(C)
    M = GSO.Mat(X, float_type="ld",
                U=IntegerMatrix.identity(d),
                UinvT=IntegerMatrix.identity(d))
    lll = LLL.Reduction(M)
    bkz = BKZ2(lll)
    g6k = Siever(M)

    for bs in range(5, beta_BKZ + 1):
        param = BKZ.Param(block_size=bs, max_loops=1, auto_abort=True)
        bkz(param)
        bkz.lll_obj()

    g6k.initialize_local(0, beta_sieve - 20, beta_sieve)
    g6k(alg="gauss")
    while g6k.l > 0:
        g6k.extend_left()
        g6k(alg="gauss" if g6k.n < 50 else "hk3")

    with g6k.temp_params(saturation_ratio=.9):
        g6k(alg="gauss" if g6k.n < 50 else "hk3")

    X_ = array(matrix(X))[:beta_sieve]

    trials = 0
    FailZ, FailC, FailN = 0, 0, 0
    success = False
    y_found = None
    for vec in g6k.itervalues():
        trials += 1
        v = array(vec)
        x = v.dot(X_)

        if (x % q == 0).all():
            FailZ += 1
            continue
        if abs(int(x[z1])) != 1:
            FailC += 1
            continue

        y = complete_solution(x)
        if int(npabs(y).max()) <= beta_infty:
            success = True
            y_found = y
            break
        FailN += 1

    db_size = g6k.db_size()
    if not success:
        return {"success": False, "trials": trials, "db": db_size,
                "FailZ": FailZ, "FailC": FailC, "FailN": FailN,
                "max_abs": None}

    f = - int(y_found[n])
    xs = vector(ZZ, list(f * y_found[:n]) + list(f * y_found[n + 1:]))
    assert xs * AISIS == u, "ISIS verification failed"
    max_abs = max(abs(int(c)) for c in xs)
    assert max_abs <= beta_infty, "solution exceeds ell_infty bound"

    return {"success": True, "trials": trials, "db": db_size,
            "FailZ": FailZ, "FailC": FailC, "FailN": FailN,
            "max_abs": max_abs}


def main():
    preset_name = "fast"
    N = 3
    if len(sys.argv) > 1:
        if sys.argv[1] not in PRESETS:
            print("Unknown preset '%s'.  Available: %s"
                  % (sys.argv[1], ", ".join(PRESETS.keys())))
            return
        preset_name = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            N = int(sys.argv[2])
        except ValueError:
            print("Usage: sage benchmark_linfty_attack.sage [fast|medium|small] [N]")
            return

    p = PRESETS[preset_name]

    print("=" * 78)
    print(" ell_infty Z-shape attack on Dilithium-flavoured ISIS")
    print(" Preset              : %s" % preset_name)
    print(" Parameters          : n=%d, m=%d, q=%d, beta_infty=%d"
          % (p["n"], 2 * p["n"], p["q"], p["beta_infty"]))
    print(" Sieve sub-block     : z1=%d, z2=%d, beta_BKZ=%d, beta_sieve=%d"
          % (p["z1"], p["z2"], p["beta_BKZ"], p["beta_sieve"]))
    print(" Predicted log2 E[#succ]/trial : %+.2f"
          % p["predicted_log2_succ"])
    print(" Trials              : %d" % N)
    print("=" * 78)

    times, succ, max_abs_list = [], 0, []

    for i in range(N):
        set_random_seed(20260424 + i)
        print("\n-- Trial %d / %d --" % (i + 1, N))
        t0 = time.time()
        try:
            res = run_one_trial(p["n"], p["q"], p["beta_infty"],
                                p["z1"], p["z2"],
                                p["beta_BKZ"], p["beta_sieve"])
        except Exception as e:
            print("  ERROR: %r" % (e,))
            continue
        dt = time.time() - t0
        times.append(dt)

        if res["success"]:
            succ += 1
            max_abs_list.append(res["max_abs"])
            print("  SUCCESS  %8.2f s   | db=%d  tried=%d  "
                  "(Z=%d, C=%d, N=%d)   ||x||_inf=%d  <=  %d"
                  % (dt, res["db"], res["trials"],
                     res["FailZ"], res["FailC"], res["FailN"],
                     res["max_abs"], p["beta_infty"]))
        else:
            print("  FAILURE  %8.2f s   | db=%d  tried=%d  "
                  "(Z=%d, C=%d, N=%d)"
                  % (dt, res["db"], res["trials"],
                     res["FailZ"], res["FailC"], res["FailN"]))

    if not times:
        print("\nNo completed trials.")
        return

    mean_t = sum(times) / len(times)
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print("   wall-clock  mean  = %8.2f s" % mean_t)
    print("               min   = %8.2f s" % min(times))
    print("               max   = %8.2f s" % max(times))
    print("   success           = %d / %d" % (succ, N))
    if max_abs_list:
        print("   solution ||.||_inf mean = %.1f   (bound = %d)"
              % (sum(max_abs_list) / len(max_abs_list), p["beta_infty"]))
        if succ > 0:
            print("   expected (s) per success = %.2f"
                  % (mean_t / (succ / N)))
    print("=" * 78)


main()
