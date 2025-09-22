from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from transformers import pipeline
from PIL import Image
from werkzeug.utils import secure_filename
import pywhatkit   # WhatsApp
import requests    # Telegram HTTP API
import telebot     # Telegram polling
import threading   # Run Telegram + Flask simultaneously

app = Flask(__name__)

load_dotenv()

# 🔹 Environment Variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

WHATSAPP_NUMBER = os.environ.get("WHATSAPP_NUMBER")       # e.g. +91xxxxxxxxxx
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 🔹 Load embeddings
embeddings = download_hugging_face_embeddings()
index_name = "healio"

# 🔹 Pinecone retriever
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 🔹 LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 🔹 Hugging Face Vision model
vision_model = pipeline("image-classification", model="google/vit-base-patch16-224")

# =====================================================
# WhatsApp Sender
# =====================================================
def send_whatsapp_message(text):
    if not WHATSAPP_NUMBER:
        print("⚠️ No WhatsApp number set in .env")
        return
    try:
        pywhatkit.sendwhatmsg_instantly(WHATSAPP_NUMBER, text)
        print(f"✅ WhatsApp message sent to {WHATSAPP_NUMBER}")
    except Exception as e:
        print(f"❌ WhatsApp send failed: {e}")

# =====================================================
# Telegram Sender (HTTP API)
# =====================================================
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing in .env")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            print("✅ Telegram message sent")
        else:
            print(f"❌ Telegram send failed: {r.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# =====================================================
# Telegram Polling Bot
# =====================================================
if TELEGRAM_BOT_TOKEN:
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

    @bot.message_handler(func=lambda message: True)
    def handle_message(message):
        user_msg = message.text
        response = rag_chain.invoke({"input": user_msg})
        answer = response.get("answer", "⚠️ No answer returned by model.")
        bot.reply_to(message, answer)

    def start_telegram_polling():
        print("🚀 Telegram bot polling started...")
        bot.polling()

    threading.Thread(target=start_telegram_polling, daemon=True).start()

# =====================================================
# Flask Routes
# =====================================================
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/ask", methods=["POST"])
def ask():
    msg = None
    image_path = None

    # JSON (text only)
    if request.is_json:
        data = request.get_json()
        msg = data.get("question", "")

    # FormData (text + optional image)
    else:
        msg = request.form.get("question", "")
        image = request.files.get("image")
        if image:
            filename = secure_filename(image.filename)
            upload_folder = "uploads"
            os.makedirs(upload_folder, exist_ok=True)
            image_path = os.path.join(upload_folder, filename)
            image.save(image_path)

    if not msg and not image_path:
        return jsonify({"answer": "⚠️ Please provide a valid question or image."})

    # Case 1: Text only
    if msg and not image_path:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "⚠️ No answer returned by model.")

    # Case 2: Image present
    else:
        img = Image.open(image_path)
        result = vision_model(img)
        condition = result[0]["label"]
        print(f"Detected condition: {condition}")

        query = f"{condition} care pathway"
        if msg:
            query = f"{condition} - {msg}"

        response = rag_chain.invoke({"input": query})
        answer = f"Detected condition: {condition} 🩺\n{response.get('answer', f'⚠️ No care pathway found for {condition}.')}"

    # Send to both WhatsApp & Telegram
    send_whatsapp_message(answer)
    send_telegram_message(answer)

    return jsonify({"answer": answer})

# =====================================================
# Main
# =====================================================
if __name__ == '__main__':
    print("🚀 Flask server starting on http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
