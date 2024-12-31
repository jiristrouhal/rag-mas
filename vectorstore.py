import dotenv


dotenv.load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings


class DocumentManager:

    def __init__(self):
        self._vectorstore = SKLearnVectorStore(
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        )
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        self._retriever = self._vectorstore.as_retriever(k=3)

    def add_documents(self, *urls: str) -> None:
        """Store document chunks and reinitialize the retriever."""
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        # Split documents
        doc_splits = self._splitter.split_documents(docs_list)
        # Add to a vector DB
        self._vectorstore.add_documents(doc_splits)
        self._reinit_retriever()

    def invoke(self, query: str) -> list[str]:
        """Retrieve documents."""
        return self._retriever.invoke(query)

    def _reinit_retriever(self):
        self._retriever = self._vectorstore.as_retriever(k=3)
