from pydantic import BaseModel

class GetInfoRequest(BaseModel):
    text: str

class GetInfoResponse(BaseModel):
    target: str
    title: str
    rating: int