import os
import telebot
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# -------------------- Load RAG --------------------
embeddings = download_hugging_face_embeddings()
index_name = "healio"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
prompt = ChatPromptTemplate.from_messages([("system", "Your system prompt here"), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -------------------- Bot --------------------
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_msg = message.text
    response = rag_chain.invoke({"input": user_msg})
    answer = response.get("answer", "⚠️ No answer returned")
    bot.reply_to(message, answer)

print("🚀 Telegram polling started...")
bot.polling(none_stop=True)
