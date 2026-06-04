from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str

class UploadResponse(BaseModel):
    rows: int
    columns: list[str]
    dtypes: dict[str, str]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str