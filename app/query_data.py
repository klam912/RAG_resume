from embedding import get_embedding_model
from create_db import CHROMA_PATH
from langchain_community.vectorstores import Chroma 
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import argparse


PROMPT_TEMPLATE = """
Answer the question based on the following context:
{context}

---
Answer the question based on the above context: {question}
"""
def query_rag(query_text: str):
    # Load the embedding model and activate the DB
    embedding = get_embedding_model()
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding
    )

    # Search the DB for 5 similar documents
    results = db.similarity_search_with_score(query_text, k=5) 

    # Create a prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Invoke the prompt in the model
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # Return the response text from the Ollama model
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Enter the query text")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


if __name__ == '__main__':
    main()
