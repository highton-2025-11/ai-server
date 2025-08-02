from pydantic import BaseModel

class ProcessContentRequest(BaseModel):
    text: str
    target: str

class ProcessContentResponse(BaseModel):
    processed_content: str