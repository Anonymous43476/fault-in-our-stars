import argparse

from sage.all import sum


def pr_rank(N, s, rank):
    H_N = sum(1 / (k ** s) for k in range(1, N + 1))
    return (1 / (H_N)) * (1 / (rank ** s))


def expected_rank_count(N, s, rank, n):
    return n * pr_rank(N, s, rank)


def get_leakage(n, N, s, t, ell, fix_leading_coeff):
    min_degree = ell - 1
    if not fix_leading_coeff:
        min_degree += 1

    leakage_count = 0

    num_reports_leaked = 0
    i = 1
    while True:
        erc = expected_rank_count(N, s, i, n)

        if erc < t and erc > min_degree:
            leakage_count += 1
            num_reports_leaked += erc

        if erc < min_degree:
            break
        i += 1

    return (leakage_count, num_reports_leaked)


def run(n, N, s, t, ell, fix_leading_coeff=False):
    leakage_count, num_reports_leaked = get_leakage(n, N, s, t, ell, fix_leading_coeff)

    print("Total Values Leaked (expected): ", leakage_count)
    try:
        print(f"Fraction of reports leaked (expected): {num_reports_leaked / n:0.2f}")
    except TypeError:
        print("(Error) Fraction of reports leaked (expected)", num_reports_leaked / n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="The number of clients", type=int, default=100_000)
    parser.add_argument("-N", help="The Zipf N-parameter", type=int, default=10_000)
    parser.add_argument("-s", help="The Zipf s-parameter", type=float, default=1.03)
    parser.add_argument("-t", help="The threshold", type=int, default=1000)
    parser.add_argument(
        "-ell", help="The degree of the polynomials", type=int, default=349
    )
    parser.add_argument(
        "--fix-leading-coeff",
        "-flc",
        help="Whether the leading coefficient of shares is fixed",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    run(args.n, args.N, args.s, args.t, args.ell, args.fix_leading_coeff)
