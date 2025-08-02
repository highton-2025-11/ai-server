from pydantic import BaseModel

class GetInfoRequest(BaseModel):
    text: str
    target: str

class GetInfoResponse(BaseModel):
    title: str
    rating: int