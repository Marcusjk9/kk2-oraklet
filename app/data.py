import logging
from io import BytesIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_dataset: pd.DataFrame | None = None


def spara_dataset(innehall: bytes) -> dict[str, Any]:
    global _dataset
    df = pd.read_csv(BytesIO(innehall))
    _dataset = df
    logger.info("Dataset laddat: %d rader, %d kolumner", len(df), len(df.columns))
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }


def hamta_statistik() -> dict[str, Any] | None:
    if _dataset is None:
        return None
    return _dataset.describe(include="all").fillna("").to_dict()


def dataset_laddat() -> bool:
    return _dataset is not None