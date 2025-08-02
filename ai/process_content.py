from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai.init import llm
from langchain.schema.runnable import RunnableMap
from openai import OpenAI

from dto.process_content import ProcessContentRequest

client = OpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "{prompt}"),
    ("human", "{raw_content}")
])

chain = RunnableMap({
    "processed_content": (prompt | llm | StrOutputParser())
})

def get_processed_data(req: ProcessContentRequest):
    return chain.invoke({"raw_content": req.text, "target": req.target, "prompt": req.instruction})