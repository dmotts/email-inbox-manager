from dotenv import find_dotenv, load_dotenv

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from custom_tools import CreateEmailDraftTool, GenerateEmailResponseTool, ReplyEmailTool, EscalateTool, CategoriseEmailTool
from fastapi import FastAPI
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

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
   # EscalateTool(),
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

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)


# Intialise FastAPI
app = FastAPI()
@app.get("/")
#def researchAgent(query: Query):
def emailInboxAgent():

    test_email="""
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

    Email 3: Newsletter Subscription
    Subject: Welcome to Our Monthly Newsletter!
    Body:
    Hi there!
    Thank you for subscribing to our monthly newsletter. Stay tuned for updates, news, and exclusive offers.
    Best,
    The [Company Name] Team

    Email 4: Spam Email
    Subject: Claim Your Free Gift Now!
    Body:
    Congratulations! You've been selected to receive a free gift. Click here to claim it now! Don’t miss out on this exclusive offer.
    Cheers,
    [Random Company]

    Email 5: Internal Update
    Subject: Staff Meeting Reminder
    Body:
    Team,
    This is a reminder about our staff meeting scheduled for tomorrow at 10 AM in the conference room. We'll be discussing quarterly goals.
    See you there,
    [Your Manager]

    Email 6: Customer Feedback Request
    Subject: We'd Love Your Feedback!
    Body:
    Dear Valued Customer,
    Thank you for your recent purchase. We hope you're loving your new product. Could you spare a few moments to provide feedback? It helps us improve.
    Thank you,
    [Your Company]

    Email 7: Important Contract Update
    Subject: Contract Amendment Required
    Body:
    Dear [Client's Name],
    Following our recent discussions, we've identified a need to amend our current contract. Please review the attached document and let us know your availability for a meeting to discuss further.
    Best,
    [Your Legal Team]
    """
    content = agent({"input": test_email})
    actual_content = content['output']
    return actual_content

