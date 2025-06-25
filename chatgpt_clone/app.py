import os
import chainlit as cl

from tools.web_search import search_web
from tools.image_gen import generate_image
from tools.call_llm import call_llm
from tools.speech_to_text import transcribe_audio
from rag.retriever import retrieve_docs, update_vector_store


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content=(
            "👋 Welcome! You can type your message or upload a document/audio file using the 📎 attachment button.\n\n"
            "**Supported formats:** `.pdf`, `.docx`, `.pptx`, `.wav`, `.mp3`, `.webm`"
        )
    ).send()


async def handle_uploaded_file(file):
    chat_history = cl.user_session.get("chat_history", [])

    ext = os.path.splitext(file.name)[1].lower()

    if ext in [".pdf", ".docx", ".pptx"]:
        status = update_vector_store(file.path)
        await cl.Message(content=f"📄 Document processed: {status}").send()

    elif ext in [".wav", ".mp3", ".webm"]:
        transcribed_text = transcribe_audio(file.path)
        if transcribed_text.strip():
            chat_history.append({"role": "user", "content": transcribed_text})
            await cl.Message(content=f"🗣️ You said: {transcribed_text}").send()

            context = [
                {"role": m["role"], "content": m["content"]}
                for m in chat_history
                if isinstance(m["content"], str)
            ]
            response = call_llm(context)
            chat_history.append({"role": "assistant", "content": response})
            await cl.Message(content=response).send()
        else:
            await cl.Message(content="❌ Sorry, I couldn't understand the audio.").send()
    else:
        await cl.Message(content="⚠️ Unsupported file type.").send()


@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")

    # ✅ Handle uploaded files first (if any)
    if message.elements:
        for element in message.elements:
            await handle_uploaded_file(element)
        return

    user_msg = message.content.strip()
    if not user_msg:
        await cl.Message(content="⚠️ I didn't catch that. Please try again.").send()
        return

    chat_history.append({"role": "user", "content": user_msg})
    message_lower = user_msg.lower()

    # 🌟 Image generation
    if "generate image" in message_lower:
        prompt = message_lower.replace("generate image", "").strip()
        image_path = generate_image(prompt)
        chat_history.append({"role": "assistant", "content": "Here's your generated image:"})
        await cl.Message(content="🖼️ Here's your generated image:").send()
        await cl.Message(content="", elements=[cl.Image(path=image_path, name="Generated Image")]).send()

    # 🔎 Web search
    elif "search" in message_lower:
        result = search_web(user_msg)
        chat_history.append({"role": "assistant", "content": result})
        await cl.Message(content=result).send()

    # 📄 Document Q&A
    elif any(k in message_lower for k in ["doc", "document", "file", "summarize", "brief", "pdf"]):
        context = retrieve_docs(user_msg)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {user_msg}\nAnswer:"
        response = call_llm([{"role": "user", "content": prompt}])
        chat_history.append({"role": "assistant", "content": response})
        await cl.Message(content=response).send()

    # 💬 General chat
    else:
        context_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history
            if isinstance(m["content"], str)
        ]
        response = call_llm(context_for_llm)
        chat_history.append({"role": "assistant", "content": response})
        await cl.Message(content=response).send()





