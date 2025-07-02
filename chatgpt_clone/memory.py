# memory.py
import os
import json

DATA_DIR = "chat_logs"
os.makedirs(DATA_DIR, exist_ok=True)

def get_filepath(user_id):
    return os.path.join(DATA_DIR, f"{user_id}.json")

def save_chat(user_id, chat_history):
    with open(get_filepath(user_id), "w") as f:
        json.dump(chat_history, f)

def load_chat(user_id):
    path = get_filepath(user_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []
