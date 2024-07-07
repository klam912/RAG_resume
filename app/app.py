from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma 
from langchain.schema.document import Document
from embedding import get_embedding_model
from text_processing import split_documents, load_documents
import os
import shutil

app = Flask(__name__)

CHROMA_PATH = '/Users/kenlam/Desktop/Data science/ML projects/RAG_resume/app/chroma'

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_model()
    )

    # Add or update the documents
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) > 0:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

@app.route('/upload-and-query', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file to a temporary directory or process it directly
        # For simplicity, assuming we save to CHROMA_PATH directly
        file.save(os.path.join(CHROMA_PATH, file.filename))

        # Process uploaded document
        # documents = []  # Assuming you need to convert the file to Document objects
        documents = load_documents(CHROMA_PATH)
        chunks = split_documents(documents)
        add_to_chroma(chunks)

        # Handle query
        query_text = request.form.get('query')
        if query_text:
            # Query processing logic (similar to previous implementation)
            # Use query_text to retrieve response from RAG app or database
            response = {"message": "Document uploaded and query processed successfully"}
            return jsonify(response), 200
        else:
            return jsonify({"error": "No query provided"}), 400


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)