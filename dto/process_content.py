from pydantic import BaseModel

class ProcessContentRequest(BaseModel):
    text: str
    target: str
    instruction: str

class ProcessContentResponse(BaseModel):
    processed_content: str