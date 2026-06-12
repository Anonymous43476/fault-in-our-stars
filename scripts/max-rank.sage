import argparse

from sage.all import *
from scipy.stats import binom


def highest_rank_with_tail(B, s, N, t, bos=40):
    """
    Return the largest rank k such that

        Pr[count(rank k) >= t] >= 2^{-bos}

    in N samples from a Zipf(B,s) distribution.
    """

    # normalization constant
    H = sum(RR(i) ** (-s) for i in range(1, B + 1))
    threshold = RR(2) ** (-bos)

    def tail_prob(k):
        p = RR(k) ** (-s) / H

        # sf(t-1) = Pr[X >= t]
        return RR(binom.sf(t - 1, N, float(p)))

    # quick checks
    if tail_prob(1) < threshold:
        return None

    if tail_prob(B) >= threshold:
        return B

    lo = 1
    hi = B

    while lo + 1 < hi:
        mid = (lo + hi) // 2

        if tail_prob(mid) >= threshold:
            lo = mid
        else:
            hi = mid

    return lo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=100_000)
    parser.add_argument("-t", type=int, default=1_000)
    args = parser.parse_args()

    B = 10_000
    s = 1.03

    k = highest_rank_with_tail(B, s, args.N, args.t)
    print("highest rank =", k)
