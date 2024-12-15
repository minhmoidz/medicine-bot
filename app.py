from flask import Flask, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import time

# Khởi tạo Flask app
app = Flask(__name__)

# Tải biến môi trường
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tải embeddings
embeddings = download_hugging_face_embeddings()

# Tạo kết nối với Pinecone
index_name = "medicinebot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Cấu hình Retriever và Model
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)
llm = OpenAI(temperature=0.4, max_tokens=500)

# Cấu hình Prompt và Chain

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# API POST nhận câu hỏi từ client và trả lời


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()

    # Lấy câu hỏi từ dữ liệu POST
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = question
        input = response
        # In ra response vào console để kiểm tra
        print("Response:", response)
        response = rag_chain.invoke({"input": question})
        # Trả về câu trả lời từ response
        if "answer" in response:
            return jsonify({"answer": response["answer"]})
        else:
            return jsonify({"error": "No answer generated"})

    except Exception as e:
        # In chi tiết thông tin lỗi
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
