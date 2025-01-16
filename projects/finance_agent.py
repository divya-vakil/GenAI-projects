#loading libraries
import phi
import os
import openai

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv

load_dotenv()

#loading keys
openai.api_key = os.environ['OPENAI_API_KEY']
phi.ap = os.environ['PHI_API_KEY']

                            
#defining webagent
web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for information",
    model = OpenAIChat(id="gpt-4o"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True
)

#defining finance agent
finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, 
                         analyst_recommendations=True, 
                         company_info=True, 
                         company_news=True, 
                         key_financial_ratios=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)


app = Playground(agents = [finance_agent, web_search_agent]).get_app()

#running app in phi
if __name__ == "__main__" :
    serve_playground_app("finance_agent:app", reload = True)
