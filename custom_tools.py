    
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
)
# CATEGORISE EMAIL
def check_consulting_email(lates_reply: str):
    prompt = f"""
    EMAIL: {lates_reply}
    ---

    Above is an email about Job offer / consulting; Your goal is identify if all information above is mentioned:
    1. What's the problem the prospect is trying to solve? 
    2. Their budget

    If all info above is collected, return YES, otherwise, return NO; (Return ONLY YES or NO)

    ANSWER: 
    """

    all_needs_collected_result = client.chat.completions.create(
        model="gpt-4",
        #model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    all_needs_collected = all_needs_collected_result.choices[0].message.content

    return all_needs_collected

def process_category(category):
    switch = {
        "URGENT":"This email requires a response",
        "ACTION_REQUIRED": "This email requires a response"
        ,
        "INFORMATIONAL": "This email does not require a response. Mark as unread",
        "NEWSLETTER": "This email does not require a response",
        "NON_REPLY": "This email has already been taken care of or replied before, nothing needs to be done now",
        "default": {
            "Step 1": "Generate email response based on guidelines",
            "Step 2": "Create email draft with the generated response"
        }
    }

    return switch.get(category, switch["default"])
    
def categorise_email(lates_reply: str):
    categorise_prompt = f"""
    EMAIL: {lates_reply}
    ---

    Your goal is to categorise the email based on categories below:
  1. URGENT: Emails that require immediate attention or response.
   2. ACTION REQUIRED: Emails that require specific actions or follow-ups.
   3. INFORMATIONAL: Emails that provide general information or updates.
   4. PERSONAL: Emails from friends, family, or personal contacts.
   5. NEWSLETTER: Emails containing newsletters or subscriptions.
   6. NON-REPLY: Auto-generated emails or offers from companies or individuals that don't require a response.
   7. OTHER: Emails that don't fit into any of the above categories. 

    CATEGORY (Return ONLY the category name in capital):
    """

    category_result = client.chat.completions.create(
        model="gpt-4",
        #model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "user", "content": categorise_prompt}
        ]
    )

    category = category_result.choices[0].message.content

    return process_category(category)  

class CategoriseEmailInput(BaseModel):
    lates_reply: str = Field(description="Latest reply from the prospect ")


class CategoriseEmailTool(BaseTool):
    name = "categorise_email"
    description = "use this to categorise email to decide what to do next"
    args_schema: Type[BaseModel] = CategoriseEmailInput

    def _run(self, lates_reply: str):
        return categorise_email(lates_reply)

    def _arun(self, url: str):
        raise NotImplementedError(
            "get_stock_performance does not support async")


# WRITE EMAIL
def generate_email_response(email_thread: str, category: str):
    # URL endpoint
    url = "https://api-bcbe5a.stack.tryrelevance.com/latest/studios/b4a2be23-4b1f-42e4-9428-5562cb997d6d/trigger_limited"

    # Headers
    headers = {
    }

    # Payload (data)
    data = {
        "params": {
            "raw_email_thread": email_thread,
            "goal": "write email response but only respond with 'Okay'",
        },
        "project": "6e075595a8e4-43eb-960d-bcad68da523e"
    }

    # Send POST request
    response = requests.post(url, headers=headers, json=data)

    return response.text


class GenerateEmailResponseInput(BaseModel):
    """Inputs for scrape_website"""
    email_thread: str = Field(description="The original full email thread")
    category: str = Field(
        description='category of email, can ONLY be "CONSULTING FOL55LOW UP" or "OTHER" ')


class GenerateEmailResponseTool(BaseTool):
    name = "generate_email_response"
    description = "use this to generate the email response based on specific guidelines, voice & tone & knowledge for the company"
    args_schema: Type[BaseModel] = GenerateEmailResponseInput

    def _run(self, email_thread: str, category: str):
        return generate_email_response(email_thread, category)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")

# ESCALATE

def escalate(original_email_address: str, message: str, additional_context: str):
    # URL to send the POST request to
    url = 'https://hooks.zapier.com/hooks/catch/15616669/38qwq19/'

    # Data to send in the POST request
    data = {
        "prospect email": original_email_address,
        "prospect message": message,
        "additional context": additional_context
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        return ('This email has been escalated to Jason, he will take care of it from here, nothing needs to be done now')
    else:
        return ('Failed to send POST request:', response.status_code)


class EscalateInput(BaseModel):
    """Inputs for scrape_website"""
    message: str = Field(
        description="The original email thread & message that was received, cc the original copy for escalation")
    original_email_address: str = Field(
        description="The email address that sent the message/email")
    additional_context: str = Field(
        description="additional context about the prospect, can be the company/prospct background research OR the consulting request details like use case, budget, etc.")


class EscalateTool(BaseTool):
    name = "escalate_to_jason"
    description = "useful when you need to escalate the case to jason or others, passing both message and original_email_address to the function"
    args_schema: Type[BaseModel] = EscalateInput

    def _run(self, original_email_address: str, message: str, additional_context: str):
        return escalate(original_email_address, message, additional_context)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")


# REPLY EMAIL
def reply_email(message: str, email_address: str, subject: str):
    return f"An email has been sent to {email_address}"

    # URL to send the POST request to
    url = 'https://hooks.zapier.com/hooks/catch/15616669/38qaaau/'

    # Data to send in the POST request
    data = {
        "Email": email_address,
        "Subject": subject,
        "Reply": message
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        return ('Email reply has been created successfully')
    else:
        return ('Failed to send POST request:', response.status_code)


class ReplyEmailInput(BaseModel):
    """Inputs for scrape_website"""
    message: str = Field(
        description="The generated response message to be sent to the email address")
    email_address: str = Field(
        description="Destination email address to send email to")
    subject: str = Field(description="subject of the email")


class ReplyEmailTool(BaseTool):
    name = "reply_email"
    description = "use this to send emails"
    args_schema: Type[BaseModel] = ReplyEmailInput

    def _run(self, message: str, email_address: str, subject: str):
        return reply_email(message, email_address, subject)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")


# CREATE EMAIL DRAFT
def create_email_draft(prospect_email_address: str, subject: str, generated_reply: str):
    # URL to send the POST request to
    url = 'https://hook.us1.make.com/rr5iyp2c56sg59rpsljvhetyndtsfdnn'

    # Data to send in the POST request
    data = {
        "email": prospect_email_address,
        "subject": subject,
        "reply": generated_reply
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Check the response
    if response.status_code == 200:
        return ('Email draft has been created successfully')
    else:
        return ('Failed to send POST request:', response.status_code)


class CreateEmailDraftInput(BaseModel):
    """Inputs for scrape_website"""
    prospect_email_address: str = Field(
        description="The prospect's email address")
    subject: str = Field(description="The original email subject")
    generated_reply: str = Field(
        description="Generated email reply to prospect")


class CreateEmailDraftTool(BaseTool):
    name = "create_email_draft"
    description = "use this to create email draft for jason to review & send"
    args_schema: Type[BaseModel] = CreateEmailDraftInput

    def _run(self, prospect_email_address: str, subject: str, generated_reply: str):
        return create_email_draft(prospect_email_address, subject, generated_reply)

    def _arun(self, url: str):
        raise NotImplementedError("failed to escalate")
