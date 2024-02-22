from dotenv import find_dotenv, load_dotenv

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from custom_tools import CreateEmailDraftTool, GenerateEmailResponseTool, ReplyEmailTool, EscalateTool, ProspectResearchTool, CategoriseEmailTool
from fastapi import FastAPI

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

system_message = SystemMessage(
    content="""
    You are an email inbox assistant of a Social Media Marketing Agency.
     
    Your goal is to handle all the incoming emails by categorising them based on 
    guideline and decide on next steps
    """
)

tools = [
    CategoriseEmailTool(),
    ProspectResearchTool(),
    EscalateTool(),
    ReplyEmailTool(),
    CreateEmailDraftTool(),
    GenerateEmailResponseTool(),
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# Intialise FastAPI
app = FastAPI()
@app.get("/")
#def researchAgent(query: Query):
def emailInboxAgent():

    test_email = """
Email 1: Important Client Inquiry
Subject: Urgent: Project Timeline Discussion Needed
Body:
Dear [Client's Name],
I hope this message finds you well. I'm reaching out to discuss our project timeline. Given the recent updates, it seems we might need to adjust our milestones. Could we schedule a call this week to go over the details?
Best,
[Your Name]

Email 2: General Inquiry
Subject: Inquiry About Your Services
Body:
Hello,
I came across your company online and am interested in learning more about your services. Could you please provide more information or a brochure? Thank you!
Kind regards,
[Name]

"""
    content = agent({"input": test_email})
    actual_content = content['output']
    return actual_content

