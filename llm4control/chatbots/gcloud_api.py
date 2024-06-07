import logging
import os
from typing import Any, Dict

import vertexai
from dotenv import load_dotenv
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import ChatSession, GenerativeModel

load_dotenv()

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def initiate_chat(
    location: str = "us-central1",
    project_id: str = os.environ.get("GCP_PROJECT_ID", ""),
    model_version: str = "gemini-1.0-pro",
) -> ChatSession:
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_version)
    chat = model.start_chat()
    return chat


def get_chat_response(chat: ChatSession, prompt: str) -> Any:
    response = chat.send_message(prompt)
    return response.text


def query_gcloud(system_prompt: str, user_prompt: str, model_version: str) -> Any:
    """
    Queries the Vertex AI model returns the response.

    :param system_prompt: A string containing the system-defined prompt.
    :param user_prompt: A string containing the user-defined prompt.
    :param model_version: A string specifying the model version to use.
    :return: The model's response as a string.
    """
    vertexai.init(project="391151572930", location="us-central1")
    parameters: Dict[str, Any] = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
    }

    if model_version == "distil":
        model_version = "842100562550849536"
        model_type = "text-bison@002"
    elif model_version == "tune":
        model_version = "8261780948643741696"
        model_type = "text-bison@001"
    else:
        model_type = "text-bison@002"

    model = TextGenerationModel.from_pretrained(model_type)

    # model = model.get_tuned_model(f"projects/391151572930/locations/
    # us-central1/models/{model_version}")

    full_prompt = system_prompt + user_prompt
    response = model.predict(full_prompt, **parameters)

    return response.text


def query_gemini(system_prompt: str, user_prompt: str, model_version: str) -> Any:
    logging.info("HTTP Request: POST to Gemini API")
    chat = initiate_chat(model_version=model_version)

    prompt = system_prompt + user_prompt
    response = get_chat_response(chat, prompt)
    logging.info(f"Response from Gemini: {response}")
    return response
