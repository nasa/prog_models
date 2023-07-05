import pandas as pd


class ProgPyDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ProgPyDataFrame

    def timestamps(self, times: list[float]=None):
        self.insert(0, 'time', times)
        self.set_index('time', inplace=True, drop=True)

    def add_timestamp(self, time: float=None, data=None):
        self.loc[time] = data
