import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document


dotenv.load_dotenv()


class DocumentManager:

    def __init__(self):
        self._vectorstore = SKLearnVectorStore(
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        )
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        self._retriever = self._vectorstore.as_retriever(k=3)

    def add_web_documents(self, *urls: str) -> None:
        """Store document chunks and reinitialize the retriever."""
        # Load documents
        docs: list[Document] = []
        for url in urls:
            try:
                docs.append(WebBaseLoader(url).load())
            except Exception as e:
                print(f"Failed to load web document from {url}: {e}")
        docs_list = [item for sublist in docs for item in sublist]
        # Split documents
        doc_splits = self._splitter.split_documents(docs_list)
        # Add to a vector DB
        self._vectorstore.add_documents(doc_splits)
        self._reinit_retriever()

    def invoke(self, query: str) -> list[str]:
        """Retrieve documents."""
        try:
            return self._retriever.invoke(query)
        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            return []

    def _reinit_retriever(self):
        self._retriever = self._vectorstore.as_retriever(k=3)
