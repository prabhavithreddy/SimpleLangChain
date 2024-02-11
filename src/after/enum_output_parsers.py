from enum import Enum


class OutputParsers(Enum):
    Pydantic = 1
    Json = 2
    PandasDataFrame = 3
    Datetime = 4