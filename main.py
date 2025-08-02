import dotenv
dotenv.load_dotenv()

from ai.translate import get_message
from ai.process_audio import get_processed_data

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/test")
def translate_english_to_italian(text: str):
    return {
        "result": get_message(text)
    }

@app.post("/process")
def process_audio(audio_file: UploadFile = File(...)):
    return get_processed_data(audio_file)