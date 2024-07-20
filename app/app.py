from flask import Flask, request, jsonify
from embedding import get_embedding_model
from langchain_community.llms import Ollama
import os
from utils import split_documents, add_to_chroma, retriever_generator, customize_rag_prompt, customize_rag_chain, get_answer
from langchain_chroma import Chroma





app = Flask(__name__)

llm = Ollama(model = "mistral")
embedding = get_embedding_model()
# Ensure upload folder exists 
UPLOAD_FOLDER = 'upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# DB path
DB_PATH = 'db'

@app.route("/chat", methods=["POST"])
def chat():
    json_content = request.json
    # Get the query
    query = json_content.get("query")

    print("Loading the vectorstore")
    # Load the vectorstore from 'db' folder
    vectorstore = Chroma(persist_directory = DB_PATH,
                         embedding_function = embedding)
    print("Successfully loading the vectorstore")

    # Creating a retriever
    print("Creating a retriever")
    retriever = retriever_generator(vectorstore)
    
    # Create a prompt
    prompt = customize_rag_prompt()

    # Create a RAG chain
    rag_chain = customize_rag_chain(prompt, llm, retriever)
    
    # Get the answer
    answer = get_answer(rag_chain, query)

    response_json = {"answer": answer}
    return response_json

@app.route("/upload", methods = ["POST"])
def upload():
    try:
        # Check if file part is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the reqest"}), 400
        
        file = request.files["file"]
        
        # Check if file is uploaded
        if file.filename == "":
            return jsonify({"error": "No file is selected"}), 400
        
        # Create a join path for the file
        save_file = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save the file in a folder
        file.save(save_file)
        print(f"File {file.filename} is saved successfully")

        # Intialize a loader
        loader = PyPDFDirectoryLoader(UPLOAD_FOLDER)
        # Load all docs in
        docs = loader.load()
        # Split the docs into chunks
        split_docs = split_documents(docs)
        # Create a vector stores to store all the chunks
        vectorstore = add_to_chroma(split_docs, embedding, DB_PATH)

        success_message = {
            "status": "File uploaded and processed successfully",
            "filename": file.filename,
            "length of chunks": len(split_docs)
        }
        return success_message

    except Exception as e:
        print(f"An error has occured: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)