import os.path

import pandas as pd
import pyarrow
from pyarrow import parquet


def get_testdata_path(filename: str) -> str:
    abs_filenpath = os.path.join(os.path.dirname(__file__), "testdata", filename)
    if not os.path.exists(abs_filenpath):
        raise ValueError(f"Cannot find {filename} from {abs_filenpath}")

    return abs_filenpath


def make_parquet_file(data_: dict | list[dict], filepath: str):
    df = pd.DataFrame(data=data_)
    table = pyarrow.Table.from_pandas(df)
    parquet.write_table(table, filepath)
