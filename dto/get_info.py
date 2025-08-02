from pydantic import BaseModel

class GetInfoRequest(BaseModel):
    text: str