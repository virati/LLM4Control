import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def query_databricks(system_prompt: str, user_prompt: str, model: str) -> str:

    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
    DATABRICKS_ENDPOINT = os.environ.get("DATABRICKS_ENDPOINT")
    if model == "mixtral":
        model = os.environ.get("MIXTRAL_MODEL")
    elif model == "llama":
        model = os.environ.get("LLAMA_MODEL")
    elif model == "dbrx":
        model = os.environ.get("DBRX_MODEL")

    client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=DATABRICKS_ENDPOINT)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": system_prompt + user_prompt},
        ],
        model=model,
        max_tokens=256,
    )

    response_text = chat_completion.choices[0].message.content
    return response_text  # type: ignore
