from pydantic import BaseModel

from superlinked.framework.common.schema.id_schema_object import IdField
from superlinked.framework.common.schema.schema_object import SchemaObject
from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.schema_object import Array, Float, Timestamp
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.space.custom_space import CustomSpace
from superlinked.framework.dsl.query.param import Param
from superlinked.framework.dsl.query.result import Result
from superlinked.framework.dsl.space.number_space import NumberSpace, Mode
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import (
    InMemoryExecutor,
)
from superlinked.framework.dsl.space.recency_space import RecencySpace
from superlinked.framework.common.dag.period_time import PeriodTime
from superlinked.framework.dsl.source.in_memory_source import InMemorySource
from superlinked.framework.dsl.query.query import Query

from .config import settings


class PhotoQueryParams(BaseModel):
    features: list[float]
    brightness: float | None = None
    features_weight: float = 1.0
    brightness_weight: float = 0.0
    recency_weight: float = 0.0


class SLClient:

    def __init__(self, embedding_size: int, period_time_list: list[PeriodTime] = None):

        self.limit = settings.superlinked.limit
        self.schema = self._generate_photo_schema()

        self.features_space = CustomSpace(
            vector=self.schema.features, length=embedding_size
        )

        max_brightness = settings.max_brightness
        self.brightness_space = NumberSpace(
            number=self.schema.brightness,
            min_value=0,
            max_value=max_brightness,
            mode=Mode.SIMILAR,
        )

        self.recency_space = RecencySpace(
            timestamp=self.schema.creation_timestamp, period_time_list=period_time_list
        )

        self.spaces = [
            self.features_space,
            self.brightness_space,
            self.recency_space,
        ]

        self.index = Index(self.spaces)
        self.source = InMemorySource(self.schema)
        self.executor = InMemoryExecutor(sources=[self.source], indices=[self.index])
        self.app = self.executor.run()

        self._query_caption = self._generate_query_caption()
        self._query_full = self._generate_query_full()

    @staticmethod
    def _generate_photo_schema() -> SchemaObject:

        @schema
        class Photo:
            filename: IdField
            brightness: Float
            features: Array
            creation_timestamp: Timestamp

        return Photo()

    def _generate_query_caption(self) -> Query:
        query = (
            Query(self.index)
            .find(self.schema)
            .similar(self.features_space.vector, Param("features"))
            .limit(self.limit)
        )
        return query

    def _generate_query_full(self) -> Query:
        query = (
            Query(
                self.index,
                weights={
                    self.features_space: Param("features_weight"),
                    self.brightness_space: Param("brightness_weight"),
                    self.recency_space: Param("recency_weight"),
                },
            )
            .find(self.schema)
            .similar(self.features_space.vector, Param("features"))
            .similar(self.brightness_space.number, Param("brightness"))
            .limit(self.limit)
        )
        return query

    def put(self, data: list):
        self.source.put(data)

    def query_full(
        self,
        params: PhotoQueryParams,
    ) -> Result:

        result = self.app.query(
            self._query_full,
            features=params.features,
            brightness=params.brightness,
            features_weight=params.features_weight,
            brightness_weight=params.brightness_weight,
            recency_weight=params.recency_weight,
        )

        return result
