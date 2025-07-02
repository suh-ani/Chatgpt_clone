import os
import chainlit as cl
import asyncio

from tools.web_search import search_web
from tools.image_gen import generate_image
from tools.stream_llm import stream_llm
from tools.speech_to_text import transcribe_audio
from rag.retriever import retrieve_docs, update_vector_store
from memory import load_chat, save_chat


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("awaiting_id", True)
    await cl.Message(content="👋 Hi! Please enter your user ID to begin (e.g., your name):").send()


async def handle_uploaded_file(file):
    ext = os.path.splitext(file.name)[1].lower()

    if ext in [".pdf", ".docx", ".pptx"]:
        await cl.Message(content=f"📁 Uploading `{file.name}` for indexing...").send()
        status = update_vector_store(file.path)
        await cl.Message(content=status).send()
        return None

    elif ext in [".wav", ".mp3", ".webm"]:
        await cl.Message(content="🗣️ Transcribing audio...").send()
        transcribed_text = transcribe_audio(file.path)
        if transcribed_text.strip():
            return f"🗣️ You said: {transcribed_text}", transcribed_text
        else:
            return "❌ Sorry, I couldn't understand the audio.", None

    else:
        return "⚠️ Unsupported file type.", None


@cl.on_message
async def on_message(message: cl.Message):
    # Step 1: Ask for user ID
    if cl.user_session.get("awaiting_id", False):
        user_id = message.content.strip().lower()
        cl.user_session.set("user_id", user_id)
        cl.user_session.set("awaiting_id", False)

        chat_history = load_chat(user_id)
        cl.user_session.set("chat_history", chat_history)

        await cl.Message(
            content=f"✅ User ID set to `{user_id}`. {'Loaded previous chat history.' if chat_history else 'Starting fresh.'}"
        ).send()
        return

    # Step 2: Conversation flow
    chat_history = cl.user_session.get("chat_history", [])
    user_id = cl.user_session.get("user_id", "default_user")

    user_msg = message.content.strip()
    has_text = bool(user_msg)
    has_files = bool(message.elements)

    if has_files:
        for element in message.elements:
            file_result = await handle_uploaded_file(element)
            if isinstance(file_result, tuple):
                status_msg, new_text = file_result
                await cl.Message(content=status_msg).send()
                if new_text:
                    user_msg = new_text
                    has_text = True
            elif file_result is not None:
                await cl.Message(content=file_result).send()

    if has_text:
        chat_history.append({"role": "user", "content": user_msg})
        message_lower = user_msg.lower()

        if "generate image" in message_lower:
            prompt = message_lower.replace("generate image", "").strip()
            image_path = generate_image(prompt)
            chat_history.append({"role": "assistant", "content": "Here's your generated image:"})
            await cl.Message(content="🖼️ Here's your generated image:").send()
            await cl.Message(content="", elements=[cl.Image(path=image_path, name="Generated Image")]).send()

        elif "search" in message_lower:
            result = search_web(user_msg)
            chat_history.append({"role": "assistant", "content": result})
            await cl.Message(content=result).send()

        elif any(k in message_lower for k in ["doc", "document", "file", "summarize", "brief", "pdf"]):
            await cl.Message(content="🔍 Retrieving relevant document sections...").send()
            max_context_tokens = 2300
            context = retrieve_docs(user_msg, max_tokens_context=max_context_tokens)

            if "⚠️ Please upload a document first." in context or "❌" in context:
                await cl.Message(content=context).send()
                return

            await cl.Message(content="💡 Document sections retrieved. Summarizing...").send()
            # Enforce max context length (e.g. 2300 tokens)
            

            # Build prompt with bounded answer space
            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {user_msg}\nAnswer:"
            response = await stream_llm([{"role": "user", "content": prompt}], max_tokens=400)

            chat_history.append({"role": "assistant", "content": response})
            

        else:
            context_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in chat_history if isinstance(m["content"], str)
            ]
            
            response = await stream_llm(context_for_llm)
            chat_history.append({"role": "assistant", "content": response})
            

    # Save chat state
    save_chat(user_id, chat_history)
    cl.user_session.set("chat_history", chat_history)


        
    
