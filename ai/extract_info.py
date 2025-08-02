from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai.init import llm
from langchain.schema.runnable import RunnableLambda, RunnableMap
from openai import OpenAI
from dto.get_info import GetInfoRequest

client = OpenAI()

# 제목 생성 프롬프트
title_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 대화 제목 생성 전문가입니다.
        사용자가 {target}에게 전달하려는 메시지를 제공합니다.
        제공된 메시지에서 사용자가 전하고 싶어하는 내용을 {target}이(가) 단번에 알아볼 수 있도록 하는 제목을 16자 이내로 출력하세요.

        규칙:
        - 메시지를 분석해 대화의 전체적인 제목을 출력한다.
        - 제목을 16자 이내로 짧게 출력한다.
        - 불필요한 추가 설명 없이 제목만 출력한다.
    """),
    ("human", "{raw_content}")
])

# 건강한 대화 점수 프롬프트
rating_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 대화 분석 전문가입니다.
        사용자가 전달한 메시지가 얼마나 건강한 대화인지 1~5점으로 평가하세요.
        1점은 매우 공격적/비하적, 5점은 매우 긍정적/건설적인 대화입니다.

        규칙:
        - 오직 점수만 숫자로 출력
        - 소수점 없이 1, 2, 3, 4, 5 중 하나만 출력
    """),
    ("human", "{raw_content}")
])

# 전체 체인 구성
chain = RunnableMap({
    "title": (title_prompt | llm | StrOutputParser()),
    "rating": (rating_prompt | llm | StrOutputParser())
})

def extract_info(req: GetInfoRequest):
    return chain.invoke({"raw_content": req.text, "target": req.target})
