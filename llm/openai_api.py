import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def query_oai(
    system_prompt: str, user_prompt: str, model_version: str, b_JSON: bool = False
) -> str:
    """Call OpenAI GPT model to get the response.

    Args:
        system_prompt (str): The system's guiding prompt.
        rules_data (str): The selected rule.
        b_JSON (bool):  Whether to request JSON format.

    Returns:
        str: The response text from the model.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    api_key = OPENAI_API_KEY
    client = OpenAI(api_key=api_key)

    params = {
        "model": model_version,
        "messages": messages,
        "stream": True,
    }

    if b_JSON:
        params["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**params)  # type: ignore

    response_text = ""
    for chunk in response:
        response_text += chunk.choices[0].delta.content or ""

    return response_text
