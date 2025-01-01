from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import faiss
import numpy as np
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, BaseModel

class CustomRetriever(BaseRetriever, BaseModel):
    chunks: List[Document] = Field(default_factory=list)
    vectorstore: Any = Field(default=None)
    embedding_model: Any = Field(default=None)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_prefix = "search_query: "
        query_text = query_prefix + query
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=True)
        query_embedding = F.layer_norm(query_embedding, normalized_shape=(query_embedding.shape[1],))
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        query_embedding_np = query_embedding.cpu().detach().numpy()
        D, I = self.vectorstore.search(query_embedding_np, k=3)
        return [self.chunks[i] for i in I[0]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

def initialize_embeddings():
    print("Initializing embeddings...")
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def process_pdf(pdf_path, embedding_model):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split document into {len(chunks)} chunks")

    document_prefix = "search_document: "
    documents = [document_prefix + chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    print("Embeddings created for chunks.")

    embeddings_np = embeddings.cpu().detach().numpy()
    vectorstore = faiss.IndexFlatL2(embeddings_np.shape[1])
    vectorstore.add(embeddings_np)

    print("Vectorstore created with FAISS.")
    return chunks, vectorstore

def initialize_qa_chain(chunks, vectorstore, embedding_model):
    llm = Ollama(model="llama3", temperature=0.1)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = CustomRetriever(chunks=chunks, vectorstore=vectorstore, embedding_model=embedding_model)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return qa_chain

def chat_with_pdf():
    pdf_path = "D:/BSP/rag/books/gullivers-travels.pdf"

    print("Initializing PDF chat system...")
    embedding_model = initialize_embeddings()
    chunks, vectorstore = process_pdf(pdf_path, embedding_model)
    qa_chain = initialize_qa_chain(chunks, vectorstore, embedding_model)

    print("\nChat initialized! Type 'quit' to exit.")
    print("Ask questions about your PDF:\n")

    while True:
        question = input("\nYou: ")

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if question.strip() == "":
            continue

        try:
            result = qa_chain({"question": question})
            print("\nAssistant:", result["answer"])
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking your question again.")

if __name__ == "__main__":
    chat_with_pdf()