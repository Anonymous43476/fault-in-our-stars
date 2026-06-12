from abc import ABC, abstractmethod
import argparse
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from typing import Any, Generic, TypeVar, Optional

from tqdm import tqdm

from sage.all import GF, Integer, PolynomialRing, ceil

GFElement = Any
PolyRing = Any
Field = Any
ID = int
PRElement = Any

T = TypeVar("T")


def sample_unique_elements(base_field: Field, n: int, current_elements):
    evaluation_points = []
    for _ in range(n):
        elt = base_field.random_element()
        while elt in evaluation_points or elt in current_elements:
            elt = base_field.random_element()
        evaluation_points.append(elt)
    return evaluation_points


def sample_unique_field_elements(F: Field, n: int):
    """
    Sample n distinct elements uniformly from the finite field F.
    Returns a list.
    """
    q = F.order()

    if not (0 <= n <= q):
        raise ValueError(f"Need 0 <= n <= {q}")

    # Small sample: sample indices directly
    if n <= q // 2:
        idxs = set(random.sample(range(q), n))
        return [a for i, a in enumerate(F) if i in idxs]

    # Large sample: sample complement
    excluded = set(random.sample(range(q), q - n))
    return [a for i, a in enumerate(F) if i not in excluded]


def serialize_poly(poly, degree):
    return [int(poly[i]) for i in range(degree + 1)]


@dataclass
class Point:
    x: GFElement
    ys: list[GFElement]


@dataclass
class InstanceValues:
    base_field: Field
    pR: PolyRing
    evals_per_point: int
    honest_degree: int


class PolynomialSampleStrategy(Enum):
    MAX_COEF_RANDOM_NONZERO = "max_coef_random_nonzero"
    MAX_COEF_ONE = "max_coef_one"
    ALL_RANDOM_COEFS = "all_random_coefs"
    RANDOM_DEGREE = "random_degree"


def sample_polynomial(
    poly_ring, degree: int, strategy: PolynomialSampleStrategy
) -> PRElement:
    match strategy:
        case PolynomialSampleStrategy.MAX_COEF_RANDOM_NONZERO:
            return poly_ring.random_element(degree=degree)

        case PolynomialSampleStrategy.MAX_COEF_ONE:
            ring_var = poly_ring.gens()[0]
            return sample_polynomial(
                poly_ring, degree - 1, PolynomialSampleStrategy.ALL_RANDOM_COEFS
            ) + (ring_var**degree)

        case PolynomialSampleStrategy.ALL_RANDOM_COEFS:
            result = poly_ring(0)
            ring_var = poly_ring.gens()[0]
            for current_deg in range(degree + 1):
                result += poly_ring.base_ring().random_element() * (
                    ring_var**current_deg
                )
            return result

        case PolynomialSampleStrategy.RANDOM_DEGREE:
            return poly_ring.random_element(degree=random.randint(0, degree))


class ShareGenType(Enum):
    RANDOM = "random"
    POLY_EVAL = "poly_eval"
    NONE = "none"


@dataclass
class ShareGenValues(Generic[T]):
    points: list[Point]
    info: T


@dataclass
class ShareGenSpec(ABC, Generic[T]):
    share_gen_type = ShareGenType.NONE

    def serialize(self):
        return {
            "share_gen_type": self.share_gen_type.value,
        }

    @abstractmethod
    def get_values(
        self, iv: InstanceValues, x_coords: list[GFElement]
    ) -> ShareGenValues[T]:
        pass


@dataclass
class RandomShareGenInfo:
    pass


@dataclass
class RandomShareGenSpec(ShareGenSpec[RandomShareGenInfo]):
    share_gen_type = ShareGenType.RANDOM

    @staticmethod
    def deserialize(_):
        return RandomShareGenSpec()

    def get_values(self, iv, x_coords: list[GFElement]):
        points = [
            Point(
                x, [iv.base_field.random_element() for _ in range(iv.evals_per_point)]
            )
            for x in x_coords
        ]
        return ShareGenValues(points, RandomShareGenInfo())


@dataclass
class PolyEvalShareGenInfo:
    polynomials: list[PRElement]


@dataclass
class PolyEvalShareGenSpec(ShareGenSpec[PolyEvalShareGenInfo]):
    share_gen_type = ShareGenType.POLY_EVAL
    strategy: PolynomialSampleStrategy
    poly_degree: int

    def serialize(self):
        return {
            **super().serialize(),
            "strategy": self.strategy.value,
            "poly_degree": int(self.poly_degree),
        }

    @staticmethod
    def deserialize(data):
        return PolyEvalShareGenSpec(
            PolynomialSampleStrategy(data["strategy"]),
            data["poly_degree"],
        )

    def get_values(self, iv, x_coords):
        polynomials = [
            sample_polynomial(iv.pR, self.poly_degree, self.strategy)
            for _ in range(iv.evals_per_point)
        ]
        points = [
            Point(x, [poly(x) for poly in polynomials])
            for x in tqdm(x_coords, position=1, leave=False)
        ]
        return ShareGenValues(points, PolyEvalShareGenInfo(polynomials))


@dataclass
class DealerSpec:
    num_points: int
    dealer_id: ID
    share_gen_spec: ShareGenSpec

    def serialize(self):
        return {
            "dealer_id": int(self.dealer_id),
            "share_gen_spec": self.share_gen_spec.serialize(),
            "num_points": int(self.num_points),
        }

    @staticmethod
    def deserialize(data):
        share_gen_spec = None
        share_gen_spec_data = data["share_gen_spec"]
        share_gen_type = share_gen_spec_data["share_gen_type"]
        match share_gen_type:
            case ShareGenType.RANDOM.value:
                share_gen_spec = RandomShareGenSpec.deserialize(share_gen_spec_data)
            case ShareGenType.POLY_EVAL.value:
                share_gen_spec = PolyEvalShareGenSpec.deserialize(share_gen_spec_data)
            case _:
                raise NotImplementedError(
                    f"Unrecognized share gen type: {share_gen_type}"
                )

        return DealerSpec(data["num_points"], data["dealer_id"], share_gen_spec)


@dataclass
class Dealer:
    spec: DealerSpec
    info: Any


@dataclass
class Instance:
    values: InstanceValues
    dealers: list[Dealer]
    codeword: list[Point]
    honest_degree: int
    field_size: int
    symbol_to_dealer: list[ID]

    threshold: int | None

    n: int = field(init=False)
    present_dealers: dict[ID, list[PRElement]] = field(init=False)
    is_nice: bool = field(init=False)

    def __post_init__(self):
        self.n = len(self.codeword)
        if len(self.codeword) > 0:
            self.c = len(self.codeword[0].ys)
        else:
            self.c = -2

        if self.threshold is None:
            self.threshold = ceil(
                (self.n + (self.c * (self.honest_degree + 1))) / (self.c + 1)
            )

        self.present_dealers = {}
        for dealer in self.dealers:
            if (
                dealer.spec.share_gen_spec.share_gen_type == ShareGenType.POLY_EVAL
                and dealer.spec.share_gen_spec.poly_degree == self.honest_degree
                and dealer.spec.num_points >= self.threshold
            ):
                self.present_dealers[dealer.spec.dealer_id] = dealer.info.polynomials

    def serialize(self):
        return {
            "parameters": {
                "field_size": int(self.field_size),
                "variable": "x",
                "n": int(self.n),
                "c": int(self.c),
                "ell": int(self.honest_degree),
                "agreement": int(self.threshold),
                "is_nice": False,
            },
            "present_dealers": {
                int(dealer_id): list(
                    map(
                        lambda poly: serialize_poly(poly, self.honest_degree), poly_list
                    )
                )
                for (dealer_id, poly_list) in self.present_dealers.items()
            },
            "codeword": list(
                map(
                    lambda symbol: [int(symbol.x), list(map(int, symbol.ys))],
                    self.codeword,
                )
            ),
            "symbol_to_dealer": list(map(int, self.symbol_to_dealer)),
        }


@dataclass
class InstanceSpec:
    field_size: int
    evals_per_point: int
    dealer_specs: list[DealerSpec]
    honest_degree: int

    def __post_init__(self):
        self.dealer_specs = sorted(self.dealer_specs, key=lambda ds: ds.dealer_id)

    def serialize(self):
        return {
            "field_size": int(self.field_size),
            "evals_per_point": int(self.evals_per_point),
            "dealer_specs": [spec.serialize() for spec in self.dealer_specs],
        }

    @staticmethod
    def deserialize(data):
        ds = [DealerSpec.deserialize(d) for d in data["dealer_specs"]]
        return InstanceSpec(
            data["field_size"], data["evals_per_point"], ds, data["honest_degree"]
        )

    def create(self, threshold=None) -> Instance:
        F = GF(Integer(self.field_size))
        instance_values = InstanceValues(
            F, PolynomialRing(F, "x"), self.evals_per_point, self.honest_degree
        )

        eval_points = []
        codeword = []
        dealers = []
        symbol_to_dealer = []

        total_points = sum([spec.num_points for spec in self.dealer_specs])

        eval_points = sample_unique_field_elements(F, total_points)

        ep_index = 0
        for dealer_spec in tqdm(self.dealer_specs):
            # current_xs = sample_unique_elements(F, dealer_spec.num_points, eval_points)
            # eval_points += current_xs
            dealer_values = dealer_spec.share_gen_spec.get_values(
                instance_values,
                eval_points[ep_index : ep_index + dealer_spec.num_points],
            )
            dealers.append(Dealer(dealer_spec, dealer_values.info))
            codeword += dealer_values.points
            symbol_to_dealer += [dealer_spec.dealer_id] * dealer_spec.num_points
            ep_index += dealer_spec.num_points

        # Permute both lists
        indices = list(range(len(codeword)))
        random.shuffle(indices)
        codeword_perm = [codeword[i] for i in indices]
        symbol_to_dealer_perm = [symbol_to_dealer[i] for i in indices]

        return Instance(
            instance_values,
            dealers,
            codeword_perm,
            self.honest_degree,
            self.field_size,
            symbol_to_dealer_perm,
            threshold,
        )


def run_malicious(args):
    dealer_specs = []
    if args.dealer is not None:
        for dealer_args in args.dealer:
            match dealer_args[2]:
                case ShareGenType.RANDOM.value:
                    dealer_specs.append(
                        DealerSpec(
                            int(dealer_args[0]),
                            int(dealer_args[1]),
                            RandomShareGenSpec(),
                        )
                    )
                case ShareGenType.POLY_EVAL.value:
                    dealer_specs.append(
                        DealerSpec(
                            int(dealer_args[0]),
                            int(dealer_args[1]),
                            PolyEvalShareGenSpec(
                                PolynomialSampleStrategy(dealer_args[3]),
                                int(dealer_args[4]),
                            ),
                        )
                    )
                case _:
                    print(f"Unrecognized dealer type: {dealer_args[2]}")
                    return 1
        inst_spec = InstanceSpec(
            Integer(args.min_field).next_prime(),
            args.c,
            dealer_specs,
            args.honest_degree,
        )

        inst = inst_spec.create()
        with open(args.filename, "w+") as outfile:
            json.dump(inst.serialize(), outfile)

        if args.config is not None:
            with open(args.config, "w+") as outfile:
                json.dump({"batch_sizes": [inst.n], "c_vals": [inst.c]}, outfile)


def sample_from_vec(freqs):
    """
    Given a discrete probability distribution `freqs`, sample an item accordingly.
    1-indexed to match use in Zipfian distributions.

    Args:
        freqs (list[float]): The probability of sampling each value

    Returns:
        int: A value sampled according to the passed frequencies
    """

    rand_num = random.random()
    cumulative_sum = 0.0
    for i, freq in enumerate(freqs):
        cumulative_sum += freq
        if rand_num < cumulative_sum:
            return i + 1


def run_zipf(args):

    normalization_const = sum(
        1 / (k**args.exponent) for k in range(1, args.support + 1)
    )
    freqs = [
        (1 / (k**args.exponent)) / normalization_const
        for k in range(1, args.support + 1)
    ]

    rank_counts = Counter(sample_from_vec(freqs) for _ in tqdm(range(args.num_points)))

    dealer_specs = []
    for rank, count in rank_counts.items():
        if rank is not None:
            if count > args.degree:
                dealer_specs.append(
                    DealerSpec(
                        count,
                        rank,
                        PolyEvalShareGenSpec(
                            PolynomialSampleStrategy.MAX_COEF_ONE, args.degree
                        ),
                    )
                )
            else:
                dealer_specs.append(DealerSpec(count, rank, RandomShareGenSpec()))

    inst_spec = InstanceSpec(
        Integer(args.min_field).next_prime(), args.c, dealer_specs, args.degree
    )
    inst = inst_spec.create(args.threshold)

    with open(args.filename, "w+") as outfile:
        json.dump(inst.serialize(), outfile)

    if args.spec is not None:
        with open(args.spec, "w+") as outfile:
            json.dump(inst_spec.serialize(), outfile, indent=2)


def main():
    parser = argparse.ArgumentParser("Generate an MDSS instance")

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "filename", type=str, help="The file to write the instance to"
    )
    parser.add_argument(
        "--min-field",
        "-mf",
        help="The minimum field size. Default 100,001",
        type=int,
        default=100_001,
    )
    parser.add_argument(
        "-c", help="The MDSS c parameter. Default 200.", type=int, default=200
    )

    subparsers = parser.add_subparsers()

    malicious_parser = subparsers.add_parser(
        "malicious",
        aliases=["m"],
        help="Generate a malicious (i.e. tightly controlled) instance",
        parents=[parent_parser],
    )
    malicious_parser.set_defaults(func=run_malicious)

    malicious_parser.add_argument(
        "--honest-degree",
        "-hd",
        help="The degree of the polynomials used by honest parties. Default 10.",
        type=int,
        default=10,
    )
    malicious_parser.add_argument(
        "--dealer",
        "-d",
        nargs="+",
        action="append",
        help="(num_points, ID, share_gen_spec_type, strategy, degree)",
    )
    malicious_parser.add_argument(
        "--config", help="Write a basic config to this path", type=str
    )

    zipf_parser = subparsers.add_parser(
        "zipf",
        aliases=["z"],
        help="Generate a zipfian instance",
        parents=[parent_parser],
    )
    zipf_parser.set_defaults(func=run_zipf)

    zipf_parser.add_argument(
        "--support",
        "-u",
        help="The support of the Zipfian distribution",
        type=int,
        default=10_000,
    )
    zipf_parser.add_argument(
        "--exponent",
        "-s",
        help="The exponent of the Zipfian distribution",
        type=float,
        default=1.03,
    )
    zipf_parser.add_argument(
        "--num-points",
        "-N",
        help="The number of points in the instance",
        type=int,
        default=100000,
    )
    zipf_parser.add_argument(
        "--degree", "-d", help="The degree of the polynomials", type=int, default=350
    )
    zipf_parser.add_argument("--spec", help="The file to write the spec to", type=str)
    zipf_parser.add_argument(
        "--threshold", "-t", help="Override the default threshold", type=int
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
