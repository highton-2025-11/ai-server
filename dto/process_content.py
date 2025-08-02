from pydantic import BaseModel

class ProcessContentRequest(BaseModel):
    text: str
    target: str