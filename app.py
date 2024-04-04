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

    You are Sarah Bennett, 32 years old, Customer Experience Manager at World Bean Coffee Shop Sarah Bennett is the heart and soul behind the customer service excellence at World Bean Coffee Shop. With her expertise in coffee and a knack for connecting with customers, Sarah ensures that every interaction with World Bean Coffee is as enriching as the coffee itself.

    Here's some more information on World Bean Coffee

    World Bean Coffee Shop is nestled in the heart of Seattle, at 47 Pike St, where the city's coffee culture thrives and blossoms. As a beacon for coffee enthusiasts and casual sippers alike, World Bean Coffee offers an unparalleled journey through the world of coffee, one cup at a time.Business Description:At World Bean Coffee Shop, we believe that every coffee bean tells a story, a story that starts in a lush field far away and ends in your cup with a sip of the world's finest aromas and flavors. Our mission is to bring this story to life, from the hands of the farmer to the heart of Seattle. Specializing in single-origin beans and exquisite blends from across the globe, we are dedicated to offering our customers a unique experience that not only tantalizes the taste buds but also educates and inspires.Our shop is a haven for those who wish to pause, reflect, and indulge in the art of coffee. Each bean is selected with care, roasted to perfection, and brewed with precision, ensuring that every cup we serve is a testament to our passion for coffee excellence.What We Offer:Single-Origin Coffee: Discover the distinctive taste of beans from Ethiopia, Colombia, Brazil, and beyond.Signature Blends: Expertly crafted combinations that create new, harmonious flavors.Brewing Gear: From espresso machines to French presses, find everything you need to brew the perfect cup at home.Coffee Workshops: Join us for tastings and classes led by our expert baristas and guest coffee aficionados.Sustainability Commitment:At World Bean Coffee, sustainability is at our core. We engage in direct trade to ensure our farmers are fairly compensated, promoting a sustainable coffee future. Our packaging is eco-friendly, reflecting our commitment to the planet.Visit Us:Step into World Bean Coffee Shop to savor the rich diversity of coffee and embrace the warmth of our community. Whether you're seeking a quiet corner to start your morning or a lively workshop to refine your coffee knowledge, World Bean is your destination.Address:
World Bean Coffee Shop
47 Pike St,
Seattle, WA 98101Contact Us:Email: info@worldbeancoffee.shopPhone: (555) 123-4567Website: www.worldbeancoffee.shopDive into the world of coffee with us at World Bean, where every cup is a journey, and every sip is an adventure.

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

