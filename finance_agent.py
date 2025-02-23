
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.newspaper4k import Newspaper4k
from phi.tools.googlesearch import GoogleSearch
from phi.playground import Playground, serve_playground_app
from phi.storage.agent.sqlite import SqlAgentStorage



load_dotenv()


finance_agent = Agent(
    name="finance Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    role="get financial data",
    instructions=["use tables to display data.","if the given any input company is not a public company then print sorry the given company doesn't trade share"],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    stream=True

)

blog_summarize_agent = Agent(
    name="summarizer agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[Newspaper4k(read_article = True, include_summary =True)],
    role="Deal with url data",
    instructions=['try to give summary in form of ullet points and paragraph and keep it long'],
    storage=SqlAgentStorage(table_name="summary_agent", db_file="agents.db"),
    add_history_to_messages=True,
    stream=True
)


news_agent = Agent(
    name="news Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    role="To provide latest news to the user",
    tools=[GoogleSearch()],
    description="You are a news agent that helps users find the latest news and latest trends relted to any topic.",
    instructions=[
        "Always try to inculude the source and date"
    ],
    storage=SqlAgentStorage(table_name="news_agent", db_file="agents.db"),
    add_history_to_messages=True,
    stream=True
)



agent_team = Agent(
    name="main Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    role="To manage all the agents",
    team=[finance_agent, blog_summarize_agent, news_agent],
    instructions=["Always include sources if the data is from news_agent",
                   "Use tables to display data if it's from financial_agent",
                     "Try to write a summary if the data is from paper_summarize_agent", 
                     'if the input consist and url of a pdf print currently i am not able to process pdfs',
                     "try to remove unnecessary * from the final output"],
    storage=SqlAgentStorage(table_name="main_agent", db_file="agents.db"),
    add_history_to_messages=True,
    stream=True
)




## it can be used to test in CLI
# is_true =True
# while is_true:
#     str = input("What would to like to know ?\n")
#     if str == "Exit" or str == "exit":
#         is_true = False
#     agent_team.print_response(str,stream=True)






# it is used to host on local system
app = Playground(agents=[agent_team]).get_app()

if __name__ == "__main__":
    serve_playground_app("finance_agent:app", reload=True)