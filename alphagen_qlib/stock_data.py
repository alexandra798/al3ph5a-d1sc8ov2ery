# --- START OF FILE stock_data.py ---

from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append(f"$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
            # result.append('$close')
        else:
            result.append(feature)
    return result


def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open', '$close', '$high', '$low', '$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
            # result.append('$close')
        else:
            raise ValueError(f"feature {feature} not supported")
    return result


class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cpu'),
                 raw: bool = False,
                 qlib_path: Union[str, Dict] = "",
                 freq: str = 'day',
                 ) -> None:
        self._init_qlib(qlib_path)
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_qlib(cls, qlib_path) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri=qlib_path, region=REG_CN)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        print(f"\nDEBUG: Loading expressions: {exprs}")
        print(f"DEBUG: Instrument: {self._instrument}")


        cal: np.ndarray = D.calendar(freq=self.freq)
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        # real_end_time = cal[min(end_index + self.max_future_days,len(cal)-1)]
        real_end_time = cal[end_index + self.max_future_days]
        result = (QlibDataLoader(config=exprs, freq=self.freq)  # type: ignore
                  .load(self._instrument, real_start_time, real_end_time))

        #---
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result shape: {result.shape}")
        print(f"DEBUG: Result columns type: {type(result.columns)}")
        if hasattr(result.columns, 'levels'):
            print(f"DEBUG: MultiIndex levels: {result.columns.levels}")
        print(f"DEBUG: First few columns: {result.columns[:5].tolist()}")
        #---

        return result

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """
        MODIFIED: This method has been updated to handle sparse financial data correctly.
        The original implementation failed because it assumed the data from qlib was a dense
        rectangle, which is not true for long time periods with changing stock listings.

        The new logic is as follows:
        1. Load sparse data from qlib.
        2. Identify all unique dates, stocks, and features present in the data.
        3. Create a complete, dense MultiIndex for the columns using all combinations of features and stocks.
        4. Reindex the original DataFrame with the new dense column index, filling missing values with NaN.
        5. Sort the columns to ensure a consistent order for reshaping (by stock, then by feature).
        6. Reshape the now-dense numpy array into a 3D tensor of shape (dates, stocks, features).
        7. Transpose the last two dimensions to get the final shape (dates, features, stocks) as expected by the rest of the code.
        """
        feature_names = ['$' + f.name.lower() for f in self._features]
        if self.raw and self.freq == 'day':
            exprs = change_to_raw(feature_names)
        elif self.raw:
            exprs = change_to_raw_min(feature_names)
        else:
            exprs = feature_names

        df = self._load_exprs(exprs)
        self.df_bak = df

        # Step 2: Identify all dimensions
        dates = df.index.unique()
        
        # Check if DataFrame has MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            stock_ids = df.columns.get_level_values(1).unique() ## 修改之处
            # In case some expressions failed, we use the actual columns
            actual_features = df.columns.get_level_values(0).unique()
        else:
            # 当DataFrame只有单层columns时，说明只有一只股票
            # 或者数据格式有问题
            print(f"WARNING: DataFrame columns are not MultiIndex!")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()[:5]}")
            print(f"DataFrame index (dates): {df.index[:5].tolist()}")
    
            # 尝试从DataFrame的其他信息推断股票列表
            # 这种情况下可能需要重新组织数据
            raise ValueError("Data format error: Expected MultiIndex columns with (feature, instrument)")

        # Step 3: Create a complete column index
        new_columns = pd.MultiIndex.from_product(
            [actual_features, stock_ids],
            names=['feature', 'instrument']
        )

        # Step 4: Reindex the DataFrame to make it dense, filling missing data with NaN
        df_dense = df.reindex(columns=new_columns)

        # Step 5: Sort columns to ensure a predictable order for reshaping
        df_dense = df_dense.sort_index(axis=1, level=['instrument', 'feature'])

        # Step 6: Get values and reshape
        values = df_dense.values
        num_dates = len(dates)
        num_stocks = len(stock_ids)
        num_features = len(actual_features)

        # Reshape to (dates, stocks, features)
        values = values.reshape((num_dates, num_stocks, num_features))

        # Step 7: Transpose to (dates, features, stocks) to match original code's expectation
        values = values.transpose(0, 2, 1)

        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return self.data.shape[1]  # Changed from len(self._features) to reflect actual data shape

    @property
    def n_stocks(self) -> int:
        return self.data.shape[2]  # Changed from data.shape[-1] for clarity

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self, data: torch.Tensor, dates: pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data, data], dim=0)
        self._dates = pd.Index(self._dates.append(dates))

    def make_dataframe(
            self,
            data: Union[torch.Tensor, List[torch.Tensor]],
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)

# --- END OF FILE stock_data.py ---