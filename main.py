import dotenv

from dto.get_info import GetInfoRequest

dotenv.load_dotenv()

from ai.extract_info import extract_info
from ai.translate import get_message
from ai.process_content import get_processed_data
from dto.process_content import ProcessContentRequest

from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
def translate_english_to_italian(text: str):
    return {
        "result": get_message(text)
    }

@app.post("/process-content")
def process_audio(req: ProcessContentRequest):
    return get_processed_data(req)

@app.post("/get-info")
def get_info(req: GetInfoRequest):
    return extract_info(req)