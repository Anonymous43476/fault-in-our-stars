import argparse


def format_bytes(n: int, precision: int = 2) -> str:
    if n < 0:
        raise ValueError("Byte size must be non-negative")

    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = float(n)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.{precision}f} {unit}"
        size /= 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument(
        "--element-bytes",
        "-b",
        type=int,
        help="The number of bytes needed for each array element. Default 4 (uint32_t).",
        default=4,
    )
    parser.add_argument(
        "--precision", "-p", type=int, help="The precision of the output", default=2
    )

    args = parser.parse_args()

    array_size = ((args.c + 1) ** 2) * (args.n + 1) * args.element_bytes

    print("Array:\t\t", format_bytes(array_size, args.precision))
