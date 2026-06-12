import argparse
from dataclasses import dataclass
from tqdm import tqdm
import json

from sage.all import RealField, log, ceil

import numpy as np
import scipy.stats as stats


def optimized_sampling_heuristic(B, s, N, k, b, t_exact=1000, bits_of_security=40):
    """
    Dynamically routes to either the unconditional calculation or the
    exact-count heuristic based on the expected number of rank k items.
    """
    RF = RealField(256)
    s_RF = RF(s)

    W = sum([RF(1) / (RF(i) ** s_RF) for i in range(k, B + 1)])
    H = sum([RF(1) / (RF(i) ** s_RF) for i in range(1, B + 1)])

    p_keep = float(W / H)
    p_k = float((RF(1) / (RF(k) ** s_RF)) / H)
    p_cond = float(p_k / p_keep)

    target_tail = 2 ** (-1 * bits_of_security)

    # Calculate the expected number of rank k items in the total population
    expected_k = N * p_k

    if expected_k > t_exact:
        # ==========================================
        # 1. Unconditional Calculation (High Expected Count)
        # ==========================================
        mu_S = N * p_keep
        sigma_S = (N * p_keep * (1 - p_keep)) ** 0.5

        if (mu_S - b) > 15 * sigma_S:
            # Fast path: Assume pool is safely >= b
            low, high = 0, min(b, N)
            z_uncond = 0
            while low <= high:
                mid = (low + high) // 2
                if stats.binom.cdf(mid - 1, b, p_cond) <= target_tail:
                    z_uncond = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return z_uncond
        else:
            # Vectorized fallback
            m_vals = np.arange(b)
            prob_S_vals = stats.binom.pmf(m_vals, N, p_keep)
            prob_S_ge_b = stats.binom.sf(b - 1, N, p_keep)

            low, high = 0, min(b, N)
            z_uncond = 0
            while low <= high:
                mid = (low + high) // 2
                cdf_vals = stats.binom.cdf(mid - 1, m_vals, p_cond)
                total_prob = np.sum(
                    prob_S_vals * cdf_vals
                ) + prob_S_ge_b * stats.binom.cdf(mid - 1, b, p_cond)

                if total_prob <= target_tail:
                    z_uncond = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return z_uncond

    else:
        # ==========================================
        # 2. Exact X_k = t_exact Calculation (Low Expected Count)
        # ==========================================
        p_other_keep = (p_keep - p_k) / (1 - p_k)

        mu_other = (N - t_exact) * p_other_keep
        sigma_other = ((N - t_exact) * p_other_keep * (1 - p_other_keep)) ** 0.5

        m_min = max(0, int(mu_other - 20 * sigma_other))
        m_max = min(N - t_exact, int(mu_other + 20 * sigma_other))

        s_other_vals = np.arange(m_min, m_max + 1)
        prob_S_other = stats.binom.pmf(s_other_vals, N - t_exact, p_other_keep)

        S_vals = s_other_vals + t_exact

        mask_lt = S_vals < b
        mask_ge = ~mask_lt

        prob_S_other_lt = prob_S_other[mask_lt]
        S_ge = S_vals[mask_ge]
        prob_S_other_ge = prob_S_other[mask_ge]

        low, high = 0, min(b, t_exact)
        z_exact = 0

        while low <= high:
            mid = (low + high) // 2

            prob_lt = np.sum(prob_S_other_lt) if (mid - 1) >= t_exact else 0.0

            if len(S_ge) > 0:
                cdf_ge = stats.hypergeom.cdf(mid - 1, S_ge, t_exact, b)
                prob_ge = np.sum(prob_S_other_ge * cdf_ge)
            else:
                prob_ge = 0.0

            total_prob = prob_lt + prob_ge

            if total_prob <= target_tail:
                z_exact = mid
                low = mid + 1
            else:
                high = mid - 1

        return z_exact


def find_min_c(batch_size, rec_thr, priv_thr, max_c):
    for c in range(1, min(max_c, batch_size) + 1):
        if rec_thr >= (batch_size / (c + 1)) + ((c * (priv_thr + 1)) / (c + 1)):
            return c
    return -1


def cost(batch_size, priv_thr, c, do_quad=False):
    if do_quad:
        return c * (batch_size - priv_thr) * c * batch_size
    else:
        return c * (batch_size - priv_thr)


@dataclass
class StepConfig:
    batch_size: int
    c_val: int
    expected_cost: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-S", help="zipf support (default 10,000)", type=int, default=10_000
    )
    parser.add_argument(
        "-s", help="zipf exponent (default 1.03)", type=float, default=1.03
    )
    parser.add_argument(
        "-k", help="polynomial degree (default 350)", type=int, default=350
    )
    parser.add_argument(
        "-N", help="Total points (default 100,000)", type=int, default=100_000
    )
    parser.add_argument("--max-c", type=int, default=10**6)
    parser.add_argument("--bits-of-security", "-bos", type=int, default=40)
    parser.add_argument("--max-rank", type=int, default=12)
    parser.add_argument("-o", type=str)
    parser.add_argument("--step", type=int, default=10_000)
    parser.add_argument("-t", type=int, help="Threshold (default 1000)", default=1_000)
    parser.add_argument("--quad", action="store_true")
    args = parser.parse_args()

    batch_sizes = []
    c_vals = []

    for rank in range(1, args.max_rank + 1):

        best = None
        prev = None

        best_wins_count = 0
        got_better_at_least_once = False

        for batch_size in tqdm(range(args.step, args.N, args.step)):
            sst = optimized_sampling_heuristic(
                args.S, args.s, args.N, rank, batch_size, args.t, args.bits_of_security
            )

            min_c = find_min_c(batch_size, sst, args.k, args.max_c)
            if min_c < 0:
                continue
            current_cost = cost(batch_size, args.k, min_c, args.quad)

            if (
                best is not None
                and current_cost >= best.expected_cost
                and got_better_at_least_once
            ):
                best_wins_count += 1
            if best is None or current_cost < best.expected_cost:
                best = StepConfig(batch_size, min_c, current_cost)
                best_wins_count = 0
                got_better_at_least_once = True

            if best_wins_count > 10:
                break

        print(rank, ":", best)
        batch_sizes.append(int(best.batch_size))
        c_vals.append(int(best.c_val))

    config = {
        "batch_sizes": batch_sizes,
        "c_vals": c_vals,
        "metadata": {
            "bits_of_security": int(args.bits_of_security),
            "N": int(args.N),
            "S": int(args.S),
            "s": float(args.s),
            "k": int(args.k),
            "t": int(args.t),
            "max_c": int(args.max_c),
            "step": int(args.step),
            "cost": "quadratic" if args.quad else "linear",
            "strategy": "assume t when under threshold",
        },
    }

    if args.o is not None:
        with open(args.o, "w+") as outfile:
            json.dump(config, outfile, indent=2)
