from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai.init import llm
from fastapi import UploadFile, File
from langchain.schema.runnable import RunnableLambda, RunnableMap
from openai import OpenAI
import tempfile
import shutil

client = OpenAI()

def transcribe_audio(file: UploadFile = File(...)):
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    return transcription.text

def extract_target_from_text(text: str) -> str:
    """LLM을 이용해 텍스트에서 대상만 추출"""
    target_prompt = f"""
    다음 문장에서 화자가 대화를 하고 있는 상대방의 호칭을 분석해 출력하세요. (ex. 어머니, 아버지, 친구 등)
    다른 설명 없이 대상만 출력하세요. (대화 상대가 존재 할 때만 호칭을 출력하고 그 외엔 X를 출력하세요)
    문장: {text}
    """
    result = llm.invoke(target_prompt)
    text = result.content.strip()
    if text == "X": return "자신"
    return text

# RunnableLambda: Whisper 1번 호출 → 대상 추출 + 원문 전달
def prepare_inputs(inputs):
    transcription_text = transcribe_audio(inputs["file"])
    inputs = {
        "target": extract_target_from_text(transcription_text),
        "raw_content": transcription_text
    }
    print(inputs)
    return inputs

prepare_inputs_runnable = RunnableLambda(prepare_inputs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 대인 관계 문제 해결 전문가입니다.
        사용자가 {target}에게 전달하려는 메시지를 제공합니다.

        당신의 목표:
        - 사용자의 의도와 핵심 내용은 그대로 유지합니다.
        - 공격적이거나 부정적인 표현, 불필요한 비하, 불쾌감을 줄 수 있는 단어를 순화합니다.
        - {target}이 기분 나쁘지 않으면서도 사용자의 의견이 명확하게 전달되도록 문장을 재구성합니다.
        - 불필요한 추가 설명 없이 원본 분량과 유사하게 유지합니다.

        출력 지침:
        1. 원본 메시지의 의미와 길이를 가능한 한 동일하게 유지하세요.
        2. 순화된 메시지만 반환하세요. (추가 설명, 주석, 불릿포인트 금지)
    """),
    ("human", "{raw_content}")
])

chain = prepare_inputs_runnable | RunnableMap({
    "target": lambda x: x["target"],
    "raw_content": lambda x: x["raw_content"],
    "processed_content": (prompt | llm | StrOutputParser())
})

def get_processed_data(audio_file: UploadFile):
    return chain.invoke({"file": audio_file})