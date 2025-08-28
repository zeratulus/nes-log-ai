def ollama_response_to_dict(ai_response):
    return {
        "content": ai_response.content,
        "additional_kwargs": ai_response.additional_kwargs,
        "response_metadata": ai_response.response_metadata,
        "id": ai_response.id,
        "usage_metadata": ai_response.usage_metadata
    }