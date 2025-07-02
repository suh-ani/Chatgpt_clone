import ollama
import chainlit as cl

async def stream_llm(messages, model="llama3.2:3b", max_tokens=400):
    try:
        print("🧠 Streaming from Ollama model:", model)
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        print("📨 Last user message:", user_msgs[-1] if user_msgs else "None")

        # Create a Chainlit message for streaming
        response_msg = cl.Message(content="")
        await response_msg.send()

        full_response = ""

        # Stream the response from Ollama with token limit
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            options={"num_predict": max_tokens}
        ):
            token = chunk.get("message", {}).get("content", "")
            full_response += token
            await response_msg.stream_token(token)  # Stream token to UI

        await response_msg.update()  # Finalize message in UI
        print("\n✅ LLM streaming complete.")
        return full_response

    except Exception as e:
        print(f"❌ Ollama stream failed: {e}")
        await cl.Message(content="⚠️ Sorry, there was an error streaming the LLM response.").send()
        return ""


