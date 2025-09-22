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
import threading   # To avoid blocking Flask

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# WHATSAPP_NUMBER = os.environ.get('WHATSAPP_NUMBER')  # Optional

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

# ---------------- Emergency Help Text ----------------
EMERGENCY_HELP_TEXT = (
    "🚨 EMERGENCY HELP 🚨\n\n"
    "📞 Ambulance: 108\n"
    "📞 Fire: 101\n"
    "📞 Police: 100\n\n"
    "🏥 Nearest Hospital: City Health Center\n"
    "☎️ Hospital Helpline: +91-9876543210\n\n"
    "💡 First Aid Tips:\n"
    "1️⃣ For bleeding → Apply pressure with a clean cloth.\n"
    "2️⃣ For burns → Run under cool water for 10 minutes.\n"
    "3️⃣ For fainting → Lay person flat & loosen tight clothing.\n\n"
    "⚠️ Stay calm and call emergency services immediately!"
)

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
    
    if "emergency" in msg:
        return jsonify({"answer": EMERGENCY_HELP_TEXT})

    print("User asked:", msg)
    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "⚠️ No answer returned.")
    print("Response:", answer)

    return jsonify({"answer": answer})

# ---------------- Emergency Route ----------------
@app.route("/help", methods=["GET"])
def emergency():
    """
    Returns emergency help info.
    Can be called from frontend or via API.
    """
    return jsonify({"emergency_help": EMERGENCY_HELP_TEXT})

# ---------------- Run Flask ----------------
if __name__ == '__main__':
    print("🚀 Flask server running on http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
