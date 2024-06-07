from typing import Any

import anthropic


def query_anthropic(system_prompt: str, user_prompt: str, model_version: str) -> Any:
    """Create a chat completion request to Anthropic.

    Args:
        user_prompt (str): The user's input prompt.
        model (str): The model to use for the completion.

    Returns:
        str: The content of the response message.
    """
    prompt = system_prompt
    prompt += user_prompt
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model_version,
        max_tokens=1000,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
    )
    return [block["text"] for block in message][0]
