from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain import hub
from embedding import get_embedding_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama



DATA_PATH = '/Users/kenlam/Desktop/Data science/ML projects/RAG_resume/data'
def load_documents(directory=DATA_PATH):
    """Load documents using Langchain's PDF loader"""
    document_loader = PyPDFDirectoryLoader(directory)
    docs = document_loader.load()
    return docs

def split_documents(documents: list[Document]):
    """Splits the documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def add_to_chroma(splits = None, embedding = get_embedding_model(), persist_directory = None):
    """Create a vector store that create embedding vectors of each document split"""
    vectorstore = Chroma.from_documents(documents=splits, embedding=get_embedding_model(), persist_directory=persist_directory)
    return vectorstore
    
def retriever_generator(vectorstore):
    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": 1})
    # retriever_docs = retriever.invoke(query)
    return retriever

def customize_rag_prompt():
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

    return prompt

def customize_rag_chain(prompt, llm, retriever):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def get_answer(rag_chain, query):
    response = rag_chain.invoke({"input": query})
    return response["answer"]

def main():
    embedding_model = get_embedding_model()
    llm = Ollama(
        model = "mistral"
    )
    docs = load_documents(DATA_PATH)
    split_docs = split_documents(docs)
    vectorstore = add_to_chroma(split_docs, embedding=embedding_model)
    query = input("Please enter a query: " )
    retriever = retriever_generator(vectorstore=vectorstore)
    prompt = customize_rag_prompt()
    rag_chain = customize_rag_chain(prompt, llm, retriever)
    answer = get_answer(rag_chain=rag_chain, query=query)
    print(answer)

if __name__ == "__main__":
    main()
    