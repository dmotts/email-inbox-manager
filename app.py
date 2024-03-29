from dotenv import find_dotenv, load_dotenv

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from custom_tools import CreateEmailDraftTool, GenerateEmailResponseTool, ReplyEmailTool, EscalateTool, CategoriseEmailTool
from fastapi import FastAPI, Form
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

system_message = SystemMessage(
    content="""
    You are an email inbox assistant of a Social Media Marketing Agency.

    Act as Daley Mottley, CEO of Zesty Digital Marketing.
     
    Your goal is to handle all the incoming emails by categorising them based on 
    guideline and decide on next steps

Please note, ALL email drafts are to be written in HTML format. 
    """
)

tools = [
    CategoriseEmailTool(),
   # EscalateTool(),
    # ReplyEmailTool(),
    CreateEmailDraftTool(),
    #GenerateEmailResponseTool(),
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
@app.post("/")
def emailInboxAgent(name: str = Form(...), email: str = Form(...), subject: str = Form(...), body: str = Form(...)):
    
    print(f"Email Received from {name}")
    
    test_email = f"""
        From:  {email}
        Subject: {subject}
        Body:
        {body}
    """
    content = agent({"input": test_email})
    actual_content = content['output']
    print(actual_content)
    return actual_content

