import logging
import os
from typing import Any
from openai import AzureOpenAI

from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# configuration for Azure Mistral
MISTRAL_ENDPOINT = os.environ.get("MISTRAL_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_KEY", "")

# Configuration for Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")

GPT35_AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("GPT35_AZURE_OPENAI_DEPLOYMENT_NAME")
GPT4_AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("GPT4_AZURE_OPENAI_DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.environ.get("OPENAI_AZURE_VERSION"),
    azure_endpoint=f"https://{AZURE_OPENAI_ENDPOINT}.openai.azure.com",  # type: ignore
)


def query_azure_mistral(
    system_prompt: str, user_prompt: str, model: str = "azureai", max_tokens: int = 50
) -> Any:
    client = MistralClient(endpoint=MISTRAL_ENDPOINT, api_key=AZURE_KEY)
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    chat_response = client.chat(model=model, messages=messages, max_tokens=max_tokens)
    response_text = chat_response.choices[0].message.content
    logging.info(f"Response from Mistral: {response_text}")
    return response_text


def query_api_gpt35(system_prompt: str, user_prompt: str) -> Any:
    # Log the request being made
    logging.info(f"HTTP Request: POST to OpenAI with GPT-3.5 model")

    response = client.chat.completions.create(
        model=os.getenv("GPT35_AZURE_OPENAI_DEPLOYMENT_NAME"),  # type: ignore
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    response_text = response.choices[0].message.content
    logging.info(f"Response from GPT-3.5: {response_text}")

    return response_text


def query_api_gpt4(system_prompt: str, user_prompt: str) -> Any:
    # Log the request being made
    logging.info(f"HTTP Request: POST to OpenAI with GPT-4 model")

    print("System Prompt: ", system_prompt)
    print("User Prompt: ", user_prompt)

    response = client.chat.completions.create(
        model=os.getenv("GPT4_AZURE_OPENAI_DEPLOYMENT_NAME"),  # type: ignore
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    response_text = response.choices[0].message.content
    logging.info(f"Response from GPT-3.5: {response_text}")

    return response_text
