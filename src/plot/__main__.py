"""Generate one combined privacy/algorithm PNG and PDF per dataset."""

from ._common import plot_parser
from .plot_combined import make_plots


def main():
    parser = plot_parser(__doc__)
    args = parser.parse_args()
    try:
        for output in make_plots(args.config):
            print(output)
            print(output.with_suffix(".pdf"))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
