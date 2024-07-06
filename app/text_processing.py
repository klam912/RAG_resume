from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


DATA_PATH = '/Users/kenlam/Desktop/Data science/ML projects/RAG_resume/data'

def load_documents(directory=DATA_PATH):
    """Load documents using Langchain's PDF loader"""
    document_loader = PyPDFDirectoryLoader(directory)
    docs = document_loader.load()
    return docs

def split_documents(documents: list[Document]):
    """Splits the documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits


def main():
    docs = load_documents()
    splits = split_documents(docs)


if __name__ == "__main__":
    main()
    