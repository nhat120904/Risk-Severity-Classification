from __future__ import annotations

import io as _io
import pandas as pd


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = _io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


