"""
cost-model comparison:
       Classical [DEP23] : SIS_optimize_attack(..., inhom='specific')
       SHARP    [Ours]  : optimize_sharp(..., inhom='specific')
if a fourth integer is given, it overrides the blocksize-sweep step
(default 2). Larger steps -> less RAM, slightly less optimal blocksize.
"""

import gc
import sys

import small_qSIS
import efficient_smallqSIS
from lengthBound import length


def clear_caches():
    """drop every BoxedTheta the two engines have accumulated."""
    small_qSIS.bt_cache.clear()
    efficient_smallqSIS._bt_cache.clear()
    gc.collect()


def evaluate(d, q, scheme, b_step=2, verbose=True):
    falcon = (scheme == "falcon")
    nu = float(length(d, q, falcon=falcon))
    w = 2 * d
    h = d

    print("=" * 70)
    print(" param: n=%d   q=%d   scheme=%s   nu=%.2f   (b_step=%d)"
          % (d, q, scheme, nu, b_step))
    print("=" * 70)

    # ---- (a) Classical ----
    clear_caches()
    print("\n[a] Classical  SIS_optimize_attack(..., inhom='specific') ...")
    saved = small_qSIS.STEPS_b
    small_qSIS.STEPS_b = b_step
    try:
        b_orig, cost_orig = small_qSIS.SIS_optimize_attack(
            q, w, h, nu, otf_lift=True, inhom="specific", verbose=verbose
        )
    finally:
        small_qSIS.STEPS_b = saved
    print("    -> beta* = %s,  cost = %.3f bits" % (b_orig, cost_orig))

    # ---- (b) SHARP ----
    clear_caches()
    print("\n[b] SHARP     optimize_sharp(..., inhom='specific') ...")
    # 1D search (beta_R = beta_S); empirically the 2D optimum lies on the
    # diagonal for Falcon/Mitaka parameters (verified at n=256, q=257).
    centre = b_orig if isinstance(b_orig, int) else 60
    half = max(20, 6 * b_step)
    beta_range = range(max(30, centre - half),
                       min(w, centre + half) + 1, b_step)
    cost_sharp, bR, bS = efficient_smallqSIS.optimize_sharp(
        q, w, h, nu, otf=True, inhom="specific",
        beta_range=beta_range,
        decouple=False,             # 1D: FAST and empirically optimal
        verbose=verbose,
    )
    if bR is None:
        print("    -> no feasible (bR, bS) found in tested ranges.")
    else:
        print("    -> (bR, bS) = (%d, %d),  cost = %.3f bits"
              % (bR, bS, cost_sharp))

    clear_caches()

    print("\n" + "-" * 70)
    print(" RESULT for (n=%d, q=%d, %s):" % (d, q, scheme))
    print("    classical  : beta*  = %-4s     cost = %7.2f bits"
          % (str(b_orig), cost_orig))
    if bR is None:
        print("    sharp     :  --  no feasible point in tested ranges.")
    else:
        delta = cost_orig - cost_sharp
        print("    sharp     : (bR,bS)= (%s,%s)   cost = %7.2f bits"
              % (str(bR), str(bS), cost_sharp))
        print("    delta     : classical - sharp = %+0.2f bits" % delta)
        print("                (positive => sharp predicts cheaper attack)")
    print("=" * 70)


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        d, q, scheme, b_step = 256, 257, "falcon", 2
    elif len(args) >= 3:
        try:
            d = int(args[0])
            q = int(args[1])
            scheme = args[2].lower()
            b_step = int(args[3]) if len(args) >= 4 else 2
        except ValueError:
            print("Usage: python3 benchmark_costs.py [n q scheme [b_step]]")
            sys.exit(1)
    else:
        print("Usage: python3 benchmark_costs.py [n q scheme [b_step]]")
        sys.exit(1)

    if scheme not in ("falcon", "mitaka"):
        print("scheme must be 'falcon' or 'mitaka'")
        sys.exit(1)

    if d >= 512 and q >= 257 and b_step < 4:
        print("[warning] (n>=512 with default b_step) is RAM-hungry: "
              "BoxedTheta arrays grow as ~nu^2.")
        print("[warning] If your process gets killed, rerun with b_step >= 4.")

    evaluate(d, q, scheme, b_step=b_step, verbose=False)


if __name__ == "__main__":
    main()
