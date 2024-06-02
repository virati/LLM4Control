from llm.azure_api import query_api_gpt35

if __name__ == "main":
    response = query_api_gpt35("You are a helpful machine", "What's the capital of France")
    print(response)
