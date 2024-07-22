import time

import boto3

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_ids = [
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1",
]

def generate_conversation(model_id, system_prompts, messages):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    print(f"Generating message with model {model_id}")

    # Inference parameters to use.
    temperature = 0.5

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    # top_k = 200
    # additional_model_fields = {"top_k": top_k}

    # Send the message.
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        # additionalModelRequestFields=additional_model_fields,
    )

    # Log token usage.
    token_usage = response["usage"]
    print(f"Input tokens: {token_usage['inputTokens']}")
    print(f"Output tokens: {token_usage['outputTokens']}")
    print(f"Total tokens: {token_usage['totalTokens']}")
    print(f"Stop reason: {response['stopReason']}")

    text_response = response["output"]["message"]["content"][0]["text"]

    return text_response


def 