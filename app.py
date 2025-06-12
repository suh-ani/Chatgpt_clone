import chainlit as cl
from tools.web_search import search_web
from tools.image_gen import generate_image
from rag.retriever import retrieve_docs

@cl.on_chat_start
def start():
    cl.user_session.set("history", [])

@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content
    if "search" in query:
        result = search_web(query)
        await cl.Message(content=result).send()
    elif "generate image" in query:
        prompt = query.replace("generate image", "")
        image_path = generate_image(prompt)
        await cl.Message(content="Here's your image:", files=[cl.File(path=image_path)]).send()

    else:
        context = retrieve_docs(query)
        # Use LLM here (OpenAI / HuggingFace model)
        response = f"Using context: {context[:300]}..."
        await cl.Message(content=response).send()
        