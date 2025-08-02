from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai.init import llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following from English into Italian."),
    ("human", "{english_message}")
])

chain = prompt | llm | StrOutputParser()

def get_message(english_message):
    result = chain.invoke({"english_message": english_message})
    return result