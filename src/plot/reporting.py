"""Display names and ordering shared by plots and summary tables."""

ALGORITHM_ORDER = (
    "vpo", "bpo", "squarecb", "fastcb", "linucb", "regcb", "adacb", "supervised",
)
ALGORITHM_LABELS = {
    "vpo": "VanillaPO",
    "bpo": "BonusPO",
    "squarecb": "SquareCB",
    "fastcb": "FastCB",
    "linucb": "LinUCB",
    "regcb": "RegCB",
    "adacb": "AdaCB",
    "supervised": "Supervised",
}


def summary_row_order(row):
    """Group privacy levels, then show algorithms in the declared display order."""
    epsilon = float("inf") if row["epsilon"] is None else float(row["epsilon"])
    return (-epsilon, float(row["delta"]),
            (*ALGORITHM_ORDER, "uniform").index(row["algorithm"]), row.get("loss", ""))
