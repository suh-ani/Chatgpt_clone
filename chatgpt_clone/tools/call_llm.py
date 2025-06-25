import ollama  # Make sure you've installed it with: pip install ollama

def call_llm(message):
    # Convert OpenAI-style message list to just the last user message
    last_user_msg = ""
    for m in reversed(message):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break

    # Call the Ollama model (llama3.2:3b or another you pulled)
    response = ollama.chat(
        model="llama3.2:3b",  # Or "llama3", "mistral", etc. based on what you have
        messages=message      # Ollama supports the same message format as OpenAI
    )

    return response['message']['content']