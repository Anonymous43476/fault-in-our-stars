from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool
import json

import numpy as np
from sage.all import Integer, ceil, sum
from tqdm import tqdm


class DBS:
    def __init__(self, L):
        """
        Initialize with length L of desired bit strings and generate instance-specific random seed

        Args:
            L (int): Length of bit strings to generate
        """
        self.L = L
        # Using dictionary to cache previously generated strings
        self._cache = {}

        # Generate instance-specific random bytes
        import os

        self._instance_key = os.urandom(32)  # 256 bits of randomness

    def get_bitstring(self, n):
        """
        Get a deterministic random L-bit string for input n

        Args:
            n (int): Input integer

        Returns:
            string: Binary string of length L
        """
        if n in self._cache:
            return self._cache[n]

        # Use SHA256 for deterministic randomness
        from hashlib import sha256

        # Combine instance key with input number
        hash_input = self._instance_key + str(n).encode("utf-8")
        hash_output = sha256(hash_input).hexdigest()

        # Convert hash to binary and take first L bits
        binary = bin(Integer("0x" + hash_output))[2:]  # Remove '0b' prefix
        result = binary[: self.L].zfill(self.L)

        # Cache and return result
        self._cache[n] = result
        return result


def pr_rank(H_N, s, rank):
    return (1 / (H_N)) * (1 / (rank**s))


def gen_expected_instance(N, s, num_clients):
    H_N = sum(1 / (k**s) for k in range(1, N + 1))

    rank_counter = Counter()

    current_rank = 1
    total_made = 0
    while total_made < num_clients:
        pr = pr_rank(H_N, s, current_rank)
        expected_num = ceil(pr * num_clients)
        rank_counter.update({current_rank: expected_num})
        total_made += expected_num
        current_rank += 1
    return rank_counter


def sample_size(id_proc, ranks, probs, size, disable_tqdm=True):
    r = Counter()
    rand_src = np.random.default_rng()

    iters = range(0, size)
    if id_proc == 0:
        iters = tqdm(iters)

    for _ in tqdm(iters, disable=disable_tqdm):
        val = rand_src.choice(ranks, p=probs)
        r.update([val])

    return r


def gen_in_distr(ranks, probs, num_clients, num_procs=1):
    rank_counter = Counter()
    print("simulating...")
    list_args = []
    NUM_PROCS = num_procs
    for itr in range(NUM_PROCS):
        q = int(num_clients / NUM_PROCS)
        if itr == (NUM_PROCS - 1):
            # you need to take the end
            size = num_clients - (NUM_PROCS - 1) * q
        else:
            size = q
        list_args.append((itr, ranks.copy(), probs.copy(), size, itr == 0))
    rank_counter = Counter()
    with Pool(NUM_PROCS) as p:
        all_counts = p.starmap(sample_size, list_args)
        for count in all_counts:
            rank_counter += count
    assert rank_counter.total() == num_clients
    return rank_counter


def count_end_nodes(ttrc, ell, t):
    def _count_end_nodes(current_prefix):
        if len(current_prefix) < ell:
            num_reports = (
                ttrc[current_prefix]["total_reports"] if current_prefix in ttrc else 0
            )
            if num_reports < t:
                return 1
            else:
                lc_prefix = current_prefix + "0"
                rc_prefix = current_prefix + "1"
                return _count_end_nodes(lc_prefix) + _count_end_nodes(rc_prefix)
        else:
            return 0

    return _count_end_nodes("")


def run_simulation(n, t, N, s, ell, cache, num_procs=1):
    dbs = DBS(ell)
    rank_counter = Counter()

    normalizing_constant = sum(1.0 / (i ^ s) for i in range(1, N + 1))
    probs = [(1.0 / (k**s)) / normalizing_constant for k in range(1, N + 1)]
    ranks = np.arange(1, N + 1)

    # if "popstar-simulator" not in cache:
    #     result = gen_in_distr(ranks, probs, n, num_procs)
    #     cache["popstar-simulator"] = result
    #
    # rank_counter = cache["popstar-simulator"]

    rank_counter = gen_expected_instance(N, s, n)

    tag_to_rank_counts = {}
    for rank, count in rank_counter.items():
        tag = dbs.get_bitstring(rank)
        # we need nodes for all internals as well
        for j in range(0, 17):
            prefix = tag[:j]
            if prefix not in tag_to_rank_counts:
                tag_node = {
                    "total_reports": int(count),
                    "revealed": False,
                    "dist_inputs": {},
                }
                tag_node["dist_inputs"][rank] = int(count)
                tag_to_rank_counts[prefix] = tag_node
            else:
                tag_to_rank_counts[prefix]["total_reports"] += int(count)
                if rank not in tag_to_rank_counts[prefix]["dist_inputs"]:
                    tag_to_rank_counts[prefix]["dist_inputs"][rank] = int(count)
                else:
                    tag_to_rank_counts[prefix]["dist_inputs"][rank] += int(count)

            if tag_to_rank_counts[prefix]["total_reports"] >= t:
                tag_to_rank_counts[prefix]["revealed"] = True

    total_leaked = 0
    num_heavy_hitters = 0
    num_end_nodes = 0
    for tag, o in tag_to_rank_counts.items():
        if o["revealed"]:
            # check children
            if len(tag) != ell:
                lc = tag + "0"
                rc = tag + "1"
                if lc in tag_to_rank_counts:
                    num_ms = len(tag_to_rank_counts[lc]["dist_inputs"].keys())
                    if tag_to_rank_counts[lc]["total_reports"] < t and num_ms == 1:
                        total_leaked += 1
                if rc in tag_to_rank_counts:
                    num_ms = len(tag_to_rank_counts[rc]["dist_inputs"].keys())
                    if tag_to_rank_counts[rc]["total_reports"] < t and num_ms == 1:
                        total_leaked += 1
            else:
                # go through each output individually and see if it should have been leaked or not
                for rank_i, count_i in o["dist_inputs"].items():
                    if count_i >= t:
                        num_heavy_hitters += 1
                    else:
                        total_leaked += 1

    seen_ranks = set(rank_counter.keys())

    return (
        total_leaked,
        num_heavy_hitters,
        len(seen_ranks) - num_heavy_hitters,
        count_end_nodes(tag_to_rank_counts, ell, t),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=100_000)
    parser.add_argument("-t", type=int, default=1000)
    parser.add_argument("-ell", "-l", type=int, default=16)
    parser.add_argument("-th", type=int, default=1)
    parser.add_argument("-N", type=int, default=10_000)
    parser.add_argument("-s", type=float, default=1.03)

    args = parser.parse_args()

    cache = {}

    a, b, c, d = run_simulation(
        args.n, args.t, args.N, args.s, args.ell, cache, args.th
    )
    print("exact counts", a)
    print("num heavy hitters", b)
    print("len seen - nhh", c)
    print("end_nodes", d)
