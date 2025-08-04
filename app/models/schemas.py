from pydantic import BaseModel

class UrlInput(BaseModel):
    url: str
