from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

app = Flask(__name__)

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index configuration
index_name = "medicalbot"

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load a local LLM (e.g., Flan-T5)
model_name = "google/flan-t5-large"  # Replace with your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a Hugging Face pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,
    temperature=0.4,
    device="cpu"  # Use "cuda" if you have a GPU
)

# Wrap the pipeline in a LangChain HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    try:
        response = rag_chain.invoke({"input": msg})
        print("Bot Response:", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print("Error:", e)
        return "Sorry, an error occurred. Please try again later."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)