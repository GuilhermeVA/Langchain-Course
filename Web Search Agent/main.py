from typing import List


from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
#from langchain_tavily import TavilySearch 


load_dotenv()




class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    answer:str = Field(description="The agent's answer to the query") 
    source:List[Source] = Field(description="A list containing the sources used to generate the answer.", default_factory=list) 




tavily = TavilyClient()


@tool
def search(query:str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search
    Returns:
        The search result    
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

#llm = ChatOpenAI(model="gpt-4.1-mini")
llm = ChatOllama(model="qwen3:1.7b")
tools = [search]
agent = create_agent(model=llm,tools=tools, response_format=AgentResponse)


def main():
    print("Hello from web-search-agent!")
    result = agent.invoke({"messages":HumanMessage(content="Busque por 3 oportunidades de estágio no Rio de Janeiro, RJ em Front-End ou UI/UX")})
    print(result)


if __name__ == "__main__":
    main()
