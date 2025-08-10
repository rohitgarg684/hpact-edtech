from pydantic import BaseModel
from typing import List, Optional

class UrlInput(BaseModel):
    url: str

class MultipleUrlsInput(BaseModel):
    urls: List[str]
    batch_size: Optional[int] = 5

class SearchQuery(BaseModel):
    query: str
    k: Optional[int] = 5
    include_graph_context: Optional[bool] = False
