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
import pywhatkit   # WhatsApp messaging
import threading   # To avoid blocking Flask

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
WHATSAPP_NUMBER = os.environ.get('WHATSAPP_NUMBER')  # Add your number in .env

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------- Embeddings ----------------
embeddings = download_hugging_face_embeddings()
index_name = "healio"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------- WhatsApp sender ----------------
def send_whatsapp_message(message: str):
    """
    Sends the chatbot answer to WhatsApp using pywhatkit in background thread.
    """
    if not WHATSAPP_NUMBER:
        print("⚠️ No WhatsApp number set in .env")
        return

    def send_msg():
        try:
            pywhatkit.sendwhatmsg_instantly(WHATSAPP_NUMBER, message, wait_time=5, tab_close=True)
            print(f"✅ WhatsApp message sent to {WHATSAPP_NUMBER}")
        except Exception as e:
            print(f"❌ WhatsApp send failed: {e}")

    threading.Thread(target=send_msg, daemon=True).start()

# ---------------- Flask routes ----------------
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    msg = data.get("question", "")
    if not msg:
        return jsonify({"answer": "Please provide a valid question."})

    print("User asked:", msg)
    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "⚠️ No answer returned.")
    print("Response:", answer)

    # Send to WhatsApp asynchronously
    send_whatsapp_message(answer)

    return jsonify({"answer": answer})

# ---------------- Run Flask ----------------
if __name__ == '__main__':
    print("🚀 Flask server running on http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
