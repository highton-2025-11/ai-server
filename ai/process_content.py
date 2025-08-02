from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai.init import llm
from langchain.schema.runnable import RunnableMap
from openai import OpenAI

from dto.process_content import ProcessContentRequest

client = OpenAI()

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

chain = RunnableMap({
    "processed_content": (prompt | llm | StrOutputParser())
})

def get_processed_data(req: ProcessContentRequest):
    return chain.invoke({"raw_content": req.text, "target": req.target})