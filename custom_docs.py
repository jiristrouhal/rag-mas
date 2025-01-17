import os

import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PDFMinerLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document

from config import REQUIRED_N_OF_RELEVANT_SOURCES


dotenv.load_dotenv()


class CustomDocManager:

    def __init__(self, name: str = "custom_doc_manager"):
        self._vectorstore = SKLearnVectorStore(
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        )
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        self._retriever = self._vectorstore.as_retriever(k=REQUIRED_N_OF_RELEVANT_SOURCES)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def add_web_documents(self, *urls: str) -> None:
        """Store document chunks and reinitialize the retriever."""
        # Load documents
        docs: list[list[Document]] = []
        for url in urls:
            try:
                docs.append(WebBaseLoader(url).load())
            except Exception as e:
                print(f"{self._name}: Failed to load web document from {url}: {e}")
        docs_list = [item for sublist in docs for item in sublist]
        self._split_and_store(docs_list)

    def add_local_documents(self, *paths: str) -> None:
        """Store document chunks and reinitialize the retriever."""
        # Load documents
        docs: list[list[Document]] = []
        for path in paths:
            try:
                if os.path.isdir(path):
                    docs.append(DirectoryLoader(path).load())
                elif os.path.isfile(path):
                    docs.append(PDFMinerLoader(path).load())
                else:
                    print(f"Path {path} is not a file or directory. Skipping.")
            except Exception as e:
                print(f"Failed to load local document from {path}: {e}")
        docs_list = [item for sublist in docs for item in sublist]
        self._split_and_store(docs_list)

    def _split_and_store(self, docs_list: list[Document]) -> None:
        doc_splits = self._splitter.split_documents(docs_list)
        # Add to a vector DB
        self._vectorstore.add_documents(doc_splits)
        self._reinit_retriever()

    def invoke(self, query: str) -> list[Document]:
        """Retrieve documents."""
        try:
            return self._retriever.invoke(query)
        except Exception as e:
            print(f"{self._name}: No documents were retrieved: {e}")
            return []

    def _reinit_retriever(self):
        self._retriever = self._vectorstore.as_retriever(k=REQUIRED_N_OF_RELEVANT_SOURCES)
