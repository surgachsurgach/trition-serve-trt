from typing import List, Optional, Union

from pyspark import sql


def assert_dataframe_equals(
    dframe1: sql.DataFrame,
    dframe2: sql.DataFrame,
    order_by: Optional[Union[str, List[str]]] = None,
):
    assert sorted(dframe1.columns) == sorted(dframe2.columns)

    dframe1 = dframe1.select(sorted(dframe1.columns))
    dframe2 = dframe2.select(sorted(dframe2.columns))

    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        dframe1 = dframe1.orderBy(*order_by)
        dframe2 = dframe2.orderBy(*order_by)

    assert dframe1.collect() == dframe2.collect()
