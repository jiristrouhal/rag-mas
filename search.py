import os
import uuid
import json
from typing import Protocol

import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_chroma import Chroma


dotenv.load_dotenv()


CONCEPT_EXTRACTOR_PROMPT = """
You are a concept extractor. You task is to identify the key concept and the best language in which to search for information about the concept.

I will provide you with the question.

Think about the main concept of the question and the best language to search for information about it.

You will then return to me the key concept in the best language to search for information.

Here are few examples:
    Me: What is the capital of Portugal?
    You: capitál de Portugal

    Me: What is the typical bakery product in Czech republic?
    You: typické pečivo v České republice
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


class Invokable(Protocol):

    def invoke(self, query: str) -> list[Document]: ...


class SearchManager:

    MAX_STORED_LAST_FOUND = 10

    def __init__(self, storage_path: str):
        self._last_found: dict[str, list[Document]] = {}
        self._translator = ChatOpenAI(model="gpt-4o-mini")
        self._grader = ChatOpenAI(model="gpt-3.5-turbo")
        self._grader_json = self._grader.bind(response_format={"type": "json_object"})
        self._web_search = WebSearch()
        self._memory = SearchMemory(storage_path)

    def invoke(self, query: str) -> list[Document]:
        concept = self._extract_concept(query)
        query = f"{query}. The question relates to the concept '{concept}'."

        recalled_docs = self._invoke_source(query, self._memory)
        relevant_docs = self._filter_relevant_docs(recalled_docs, query)

        if len(relevant_docs) < 2:
            new_documents = self._invoke_source(query, self._web_search)
            self._update_last_found(query, new_documents)
            new_relevant = self._filter_relevant_docs(new_documents, query)
            self._docs_were_relevant(query, [str(doc.id) for doc in new_relevant])
            relevant_docs.extend(new_relevant)

        return relevant_docs

    def _filter_relevant_docs(self, documents: list[Document], query: str) -> list[Document]:
        relevant_docs = []
        for d in documents:
            if self._is_relevant(document=d, query=query):
                relevant_docs.append(d)
        return relevant_docs

    def _docs_were_relevant(self, query: str, doc_ids: list[str]) -> None:
        if query not in self._last_found:
            print(f"There are no search results waiting to be stored for the query '{query}'.")
        last_found = self._last_found.pop(query, [])
        relevant = []
        for d in last_found:
            doc_id = d.id
            if doc_id in doc_ids:
                relevant.append(d)
        if relevant:
            print(f"Memorizing {len(relevant)} documents for query {query}.")
            self._memory.add(relevant)

    def _update_last_found(self, query: str, documents: list[Document]) -> None:
        self._last_found[query] = documents.copy()
        over_limit = len(self._last_found) - max(self.MAX_STORED_LAST_FOUND, 1)
        if over_limit > 0:
            old_queries = list(self._last_found.keys())[:over_limit]
            for query in old_queries:
                print(f"Clearing search results to be stored for and old query '{query}'.")
                del self._last_found[query]

    def _invoke_source(self, query: str, invokable: Invokable) -> list[Document]:
        docs = invokable.invoke(query)
        docs = self._filter_out_duplicates(docs)
        return docs

    def _filter_out_duplicates(self, documents: list[Document]) -> list[Document]:
        # unique documents are keyed by the source
        unique_documents: dict[str, list[Document]] = dict()
        for doc in documents:
            source = doc.metadata["source"]
            if source in unique_documents:
                if doc.page_content not in [d.page_content for d in unique_documents[source]]:
                    unique_documents[source].append(doc)
                # Skipping duplicate document
            else:
                unique_documents[source] = [doc]
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
        return str(
            self._translator.invoke(
                [SystemMessage(CONCEPT_EXTRACTOR_PROMPT), HumanMessage(question)]
            ).content
        )


class SearchMemory:

    def __init__(self, storage_path: str, max_results: int = 3):
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        self._search_storage = Chroma(
            "search_storage",
            OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=storage_path,
        )
        self._max_results = max_results

    def add(self, documents: list[Document]) -> None:
        assert isinstance(documents, list), f"Expected list of Document, got {documents}"
        if documents:
            self._search_storage.add_documents(documents)

    def invoke(self, query: str) -> list[Document]:
        assert isinstance(query, str), f"Expected str, got {query}"
        return self._search_storage.similarity_search(query, k=self._max_results)


class WebSearch:
    """Search the web for information related to the question."""

    def __init__(self):
        self._web_search_tool = TavilySearchResults(max_results=3)

    def invoke(self, question: str) -> list[Document]:
        """Search the web for information related to the question.

        Args:
            question (str): The question to search for.
        Returns:
            list[Document]: The documents found on the web.
        """
        docs: list[dict] = self._search(question)
        documents = [self._convert_search_result_to_document(doc) for doc in docs]
        return documents

    def _search(self, query: str) -> list[dict]:
        docs: list = self._web_search_tool.invoke({"query": query})
        assert all(isinstance(doc, dict) for doc in docs), f"Expected list of Document, got {docs}"
        return docs

    def _convert_search_result_to_document(self, search_result: dict) -> Document:
        return Document(
            page_content=search_result["content"],
            id=str(uuid.uuid4()),
            metadata={"source": search_result["url"]},
        )
