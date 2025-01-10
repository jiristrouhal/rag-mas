from __future__ import annotations
import abc
import os
import uuid
import json
from typing import Literal, Type, Protocol, Optional, Any

import dotenv
from langchain_community.retrievers import (
    PubMedRetriever,
    ArxivRetriever,
    TavilySearchAPIRetriever,
    WikipediaRetriever,
)
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_chroma import Chroma


from config import REQUIRED_N_OF_RELEVANT_SOURCES


dotenv.load_dotenv()


CONCEPT_EXTRACTOR_PROMPT = """
You are a concept extractor. You task is to identify the key concept and the best language in which to search for information about the concept.

I will provide you with the question.

You will return to me the main concept, the question is about.

I will then ask you to think about in what language (for example English, Czech, ...) would be the best to search for information about the concept.

You will write to me your thoughts on the language. If you are not sure, admit it and suggest English.

I will then ask you to return the concept in the best language to search for information.

You will return the concept translated into the language from your reasoning. Return only the concept.

Here is the question:
"""


SUBCONCEPT_EXTRACTOR_PROMPT = """
You are a concept extractor. You task is to identify the key concepts related to the question of concept I provide to you.

I will provide you with a question or concept.

You will return to me a list of key concepts contained in or related to the question or concept.

Each concept must be still directly related to the question or concept.

Good example:
    I write:
        the key features of the RAG systems and their applications
    You return:
        [RAG systems, RAG applications]
Bad example (the key concepts are too general and unrelated to the main concept):
    I write:
        the key features of the RAG systems and their applications
    You return:
        [features, applications]

Return only the list, that can be parsed as a JSON array. Do not include any other information.

Here is my question or concept:
"""


DOC_GRADER_PROMPT = """

You are a grader assessing the relevance of a retrieved document to a user question.

Here is the retrieved document:
{document}

Carefully and objectively assess whether the document is relevant. Follow these rules.
- If the document addresses some of the concepts in the question, it is relevant.
- If the document contains the keyword(s) from the question, it is relevant.
- If the document contains information, that might guide me to find the information on my own, it is relevant.
Only if none of the above is true, the document is not relevant.

Then respond to me with a thought on the document's relevance. If there is some hint of relevance, conclude your answer as the document is relevant even though it does not contain all the information.
I will then ask you to grade the document's relevance.
You will answer with 'no' if the document is not relevant. Otherwise, respond with 'yes'.

You will then return JSON with a single key, binary_score, that is 'yes' or 'no'
Here our conversation ends.

Example:

User:
    Here is the retrieved document: xxx
    Here is the user question: yyy

You:
    The key concepts asked in the question are [list of single-phrase concepts]. The document adresses two of the five main concepts, which supports relevance. Altought the document does not directly address the key information asked for in the question ... and as a whole, it is relevant.

User:
    Grade the document relevance with 'yes' or 'no'.

You:
    {{"binary_score": "yes"}}

Now I will give you my question.
"""


# SearchName = Literal["generic", "medical", "scientific", "langchain_api"]
SearchName = Literal["generic", "medical", "scientific"]


class Retriever(Protocol):
    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Document]: ...

    @property
    def name(self) -> str: ...


class SearchMemory:

    def __init__(
        self, name: str, storage_path: str, max_results: int = REQUIRED_N_OF_RELEVANT_SOURCES
    ):
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        self._search_storage = Chroma(
            name + "_db",
            OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=storage_path,
        )
        self._max_results = max_results
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def add(self, documents: list[Document]) -> None:
        assert isinstance(documents, list), f"Expected list of Document, got {documents}"
        for d in documents:
            assert isinstance(d, Document), f"Expected Document, got {type(d)}"
        if documents:
            self._search_storage.add_documents(documents)

    def invoke(self, query: str, **kwargs) -> list[Document]:
        assert isinstance(query, str), f"Expected str, got {query}"
        return self._search_storage.similarity_search(query, k=self._max_results)


class Search(abc.ABC):

    DEFAULT_NAME = "search"

    def __init__(self, name: str):
        self._retriever = self._construct_retriever()
        self._name = name if name.strip() else self.DEFAULT_NAME
        assert isinstance(self._name, str), f"Expected str, got {self._name}"

    @property
    def name(self) -> str:
        return self._name

    def invoke(self, query: str) -> list[Document]:
        """Search the Arxiv for information related to a question."""
        docs: list[Document] = self._retriever.invoke(query)
        for doc in docs:
            assert isinstance(doc, Document), f"Expected Document, got {type(doc)}, {doc}"
            doc.metadata["search_tool"] = self._name
            doc.metadata["from_memory"] = False
            for key, value in doc.metadata.items():
                if type(value) not in [str, int, float, bool]:
                    doc.metadata[key] = str(value)
        docs = self._proces_docs(docs)
        for d in docs:
            assert isinstance(
                d, Document
            ), f"Search {self._name}: Expected processed Document, got {type(d)}"
        return docs

    def _proces_docs(self, docs: list) -> list[Document]:
        return docs

    @abc.abstractmethod
    def _construct_retriever(self) -> BaseRetriever | BaseTool:
        pass


class ArxivSearch(Search):
    """Search the Arxiv for information related to a question."""

    DEFAULT_NAME = "arxiv_search"

    def _construct_retriever(self) -> ArxivRetriever:
        return ArxivRetriever(top_k_results=REQUIRED_N_OF_RELEVANT_SOURCES)


class PubMedSearch(Search):
    """Search the PubMed for information related to the question."""

    DEFAULT_NAME = "pubmed_search"

    def _construct_retriever(self) -> PubMedRetriever:
        api_key = dotenv.get_key(dotenv.find_dotenv(), "PUBMED_API_KEY")
        return PubMedRetriever(top_k_results=REQUIRED_N_OF_RELEVANT_SOURCES, api_key=api_key)


class WikiSearch(Search):

    DEFAULT_NAME = "wikipedia_search"

    def _construct_retriever(self) -> WikipediaRetriever:
        return WikipediaRetriever(top_k_results=REQUIRED_N_OF_RELEVANT_SOURCES)


class WebSearch(Search):
    """Search the web for information related to the question."""

    DEFAULT_NAME = "web_search"
    INCLUDE_DOMAINS: list[str] = []

    def _construct_retriever(self) -> TavilySearchAPIRetriever:
        return TavilySearchAPIRetriever(
            k=REQUIRED_N_OF_RELEVANT_SOURCES,
            include_images=False,
            include_raw_content=True,
            include_domains=self.INCLUDE_DOMAINS,
        )

    def _proces_docs(self, docs: list[Document]) -> list[Document]:
        processed_docs = []
        for doc in docs:
            if not doc.page_content.strip():
                continue
            doc.metadata.pop("images", [])
            doc.id = str(uuid.uuid4())
            processed_docs.append(doc)
        return docs


class LangchainAPISearch(WebSearch):
    """Search the 'https://api.python.langchain.com' for information related to the question."""

    DEFAULT_NAME = "langchain_api_search"
    INCLUDE_DOMAINS = ["api.python.langchain.com"]


_SEARCH_TOOL_DICT: dict[SearchName, Type[Search]] = {
    # "langchain_api": LangchainAPISearch,
    "scientific": ArxivSearch,
    "medical": PubMedSearch,
    "generic": WikiSearch,
}


def _search_tool(search_type: SearchName, name: str) -> Search:
    types = str(tuple(_SEARCH_TOOL_DICT.keys()))[1:-1]
    if search_type not in _SEARCH_TOOL_DICT:
        raise ValueError(
            f"The search tool '{search_type}' does not exist. It must be one of the following: {types}."
        )
    return _SEARCH_TOOL_DICT[search_type](name)


class SearchManager:

    MAX_STORED_LAST_FOUND = 20

    def __init__(self, name: str, storage_path: str, search_type: SearchName = "generic"):
        self._main_concept_extractor = ChatOpenAI(model="gpt-4o-mini")
        self._subconcept_extractor = ChatOpenAI(model="gpt-4o-mini").bind(
            response_format={"type": "json_object"}
        )
        self._grader = ChatOpenAI(model="gpt-4o-mini")
        self._grader_json = self._grader.bind(response_format={"type": "json_object"})
        self._search_type = search_type
        self._search = _search_tool(search_type, name)
        assert self._search.name is not None, "Search tool name must not be empty"
        self._memory = SearchMemory(name=name.lower(), storage_path=storage_path)

    @property
    def name(self) -> str:
        return self._search.name

    def _search_for_concept(self, query: str, max_depth: int = 1) -> list[Document]:
        print(f"Recalling memories from search of type {self._search_type} for concept {query}.")
        recalled_docs = self._invoke_source(query, self._memory)
        relevant_docs = self._filter_relevant_docs(recalled_docs, query)
        if len(relevant_docs) < REQUIRED_N_OF_RELEVANT_SOURCES:
            print(f"Invoking search of type {self._search_type} for concept {query}.")
            new_documents = self._invoke_source(query, self._search)
            new_relevant = self._filter_relevant_docs(new_documents, query)
            relevant_docs.extend(new_relevant)

        if not relevant_docs and max_depth > 1:
            messages = [SystemMessage(SUBCONCEPT_EXTRACTOR_PROMPT), HumanMessage(query)]
            subconcepts = json.loads(str(self._subconcept_extractor.invoke(messages).content))
            for subconcept in subconcepts:
                subconcept_docs = self._search_for_concept(subconcept, max_depth - 1)
                relevant_docs.extend(subconcept_docs)

        return relevant_docs

    def invoke(self, query: str) -> list[Document]:
        concept = self._extract_concept(query)
        relevant_docs = self._search_for_concept(concept, max_depth=2)
        if relevant_docs:
            print(f"'{self._search.name}' has found {len(relevant_docs)} relevant documents.")
            self._memory.add(relevant_docs)
        else:
            print(f"No relevant documents found with '{self._search.name}' for query '{query}'.")
        return relevant_docs

    def _filter_relevant_docs(self, documents: list[Document], query: str) -> list[Document]:
        relevant_docs = []
        for d in documents:
            if self._is_relevant(document=d, query=query):
                relevant_docs.append(d)
        return relevant_docs

    def _invoke_source(self, query: str, invokable: Retriever) -> list[Document]:
        docs = invokable.invoke(query)
        print(f"Search tool '{invokable.name}': Retrieved {len(docs)} documents.")
        docs = self._filter_out_duplicates(docs)
        return docs

    def _filter_out_duplicates(self, documents: list[Document]) -> list[Document]:
        # unique documents are keyed by the source
        unique_documents: dict[str, list[Document]] = dict()
        for doc in documents:
            source_metadata = str(doc.metadata)
            if source_metadata in unique_documents:
                if doc.page_content not in [
                    d.page_content for d in unique_documents[source_metadata]
                ]:
                    unique_documents[source_metadata].append(doc)
                # Skipping duplicate document
            else:
                unique_documents[source_metadata] = [doc]
        return [doc for docs in unique_documents.values() for doc in docs]

    def _is_relevant(self, document: Document, query: str) -> bool:
        doc_grader_prompt_formatted = DOC_GRADER_PROMPT.format(
            document=document.page_content, question=query
        )
        messages: list[BaseMessage] = [
            SystemMessage(doc_grader_prompt_formatted),
            HumanMessage(query),
        ]
        thought = self._grader.invoke(messages)
        messages.extend([thought, HumanMessage("Grade the document relevance with 'yes' or 'no'.")])
        result = self._grader_json.invoke(messages)
        grade = str(json.loads(str(result.content))["binary_score"])
        is_relevant = grade.lower() == "yes"
        return is_relevant

    def _extract_concept(self, question: str) -> str:
        """Translate the key concept from the question to the best language to search for information.

        For example:
        Question: What is the capital of Portugal?
        Translated concept: capitál de Portugal

        Args:
            question (str): The question to translate.
        Returns:
            str: The translated concept
        """
        messages = [SystemMessage(CONCEPT_EXTRACTOR_PROMPT), HumanMessage(question)]
        concept = self._main_concept_extractor.invoke(messages)
        messages.extend([concept, HumanMessage("Think about the language for search")])
        thought = self._main_concept_extractor.invoke(messages)
        messages.extend([thought, HumanMessage("Return the concept in the best language")])
        translated_concept = str(self._main_concept_extractor.invoke(messages).content)
        return translated_concept
