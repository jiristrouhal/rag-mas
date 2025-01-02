import operator
from typing import TypedDict, Annotated, Literal
import json
import os

from IPython.display import Image
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from vectorstore import DocumentManager
from websearch import web_search_tool


DOC_GRADER_INSTRUCTIONS = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
"""


DOC_GRADER_PROMPT = """
Here is the retrieved document:
{document}

Here is the user question:
{question}

Carefully and objectively assess whether the document is relevant. The document is relevant if:
- it contains the main concept from the question.
- it contains the keyword(s) from the question.

Then respond to me with a thought on the document relevance.
I will then ask you to grade the document relevance with 'yes' or 'no'.
You will then return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains
at least some information that is at least partially relevant to the question. Here our conversation ends.

Example:

Me:
    Here is the retrieved document: xxx
    Here is the user question: yyy
You:
    The document contains information partially relevant to the question because ... and as a whole, it is relevant.
Me:
    Grade the document relevance with 'yes' or 'no'.
You:
    {{"binary_score": "yes"}}

"""


ROUTER_INSTRUCTIONS = """
You are an expert at routing a user question to a vectorstore or web search.

Follow these rules when deciding where to route the question:
- Use the vectorstore contains specific and non-changing information.
- Use web-search for current events or information.

First, return to be a reasoning about if the question should be routed to the vectorstore or web search.
Respond in a plain text of one to three sentences.

I will then ask you to choose the datasource.

You will then return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question.
Then our conversation ends.

Example:

Me: What is the current weather in New York City?
You: This question is about current events, so it should be routed to web search.
Me: Choose the datasource.
You: {"datasource": "websearch"}
"""


RAG_PROMPT = """
You are an assistant for question-answering tasks.
Here is the context to use to answer the question:
{context}

Think carefully about the above context.
Now, review the user question:
{question}

Provide an answer to this questions using only the above context.
Use three sentences maximum and keep the answer concise.

Answer:
"""


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to and modify in each graph node.
    """

    question: str  # User question
    answer: str  # Answer generated
    web_search: str  # binary decision to perform web search
    max_retries: int  # Max number of retries for answer generation
    loop_step: Annotated[int, operator.add]
    documents: list[Document]  # List of documents to search


class CitedAnswer(TypedDict):
    """
    Cited answer is a dictionary that contains information about the answer and its source.
    """

    answer: str  # Answer generated
    sources: list[Document]  # List of sources for the answer

    def __str__(self) -> str:
        return self["answer"] + "\n\n" + "\n".join(self["sources"])

    @staticmethod
    def cited_document(doc: Document) -> str:
        return f"{doc.metadata}"


class RetrieverSimple:

    def __init__(self):
        self._document_manager = DocumentManager()
        self._llm = ChatOpenAI(model="gpt-4o-mini")
        self._llm_json_mode = self._llm.bind(response_format={"type": "json_object"})
        self._websearch_tool = web_search_tool
        self._construct_graph()

    def add_web_documents(self, *urls: str) -> None:
        self._document_manager.add_web_documents(*urls)

    def invoke(self, query: str) -> str:
        graph_response = self._graph.invoke(GraphState(question=query))
        return self._format_invoke_response(graph_response)

    def _format_invoke_response(self, output_state: GraphState) -> str:
        return (
            output_state["answer"]
            + "\n\n"
            + "\n\n".join([doc.metadata["source"] for doc in output_state["documents"]])
        )

    def print_graph_png(self, path: str, name: str = "retriever") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)

    def retrieve(self, state: GraphState) -> dict:
        """
        Retrieve documents from vectorstore

        Args:
            state (GraphState): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]

        # Write retrieved documents to documents key in state
        documents = self._document_manager.invoke(question)
        return {"documents": documents}

    def generate(self, state: GraphState) -> dict:
        """Generate answer using RAG on retrieved documents."""
        question: str = state["question"]
        documents: list[Document] = state["documents"]
        loop_step: int = state.get("loop_step", 0)

        # RAG generation
        docs_txt = format_docs(documents)
        rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)
        message = self._llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"answer": message.content, "loop_step": loop_step + 1}

    def grade_documents(self, state: GraphState) -> dict:
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (GraphState): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
        question = state["question"]
        documents: list[Document] = state["documents"]

        # Score each doc
        filtered_docs = []
        # The websearch will be done if either some document is not relevant or if there are no documents
        web_search = "No" if documents else "Yes"
        for d in documents:
            doc_grader_prompt_formatted = DOC_GRADER_PROMPT.format(
                document=d.page_content, question=question
            )
            messages = [SystemMessage(content=DOC_GRADER_INSTRUCTIONS)] + [
                HumanMessage(content=doc_grader_prompt_formatted)
            ]
            thought = self._llm.invoke(messages)
            messages.extend(
                [thought, HumanMessage("Grade the document relevance with 'yes' or 'no'.")]
            )
            result = self._llm_json_mode.invoke(messages)
            grade = str(json.loads(result.content)["binary_score"])
            # Document relevant
            if grade.lower() == "yes":
                filtered_docs.append(d)
            # Document not relevant
            else:
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"

        return {"documents": filtered_docs, "web_search": web_search}

    def web_search(self, state: GraphState) -> dict:
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        docs = web_search_tool.invoke({"query": question})
        for d in docs:
            web_result = "\n".join([d["content"] for d in docs])
            web_result = Document(page_content=web_result, metadata={"source": d["url"]})
            documents.append(web_result)
        return {"documents": documents}

    def route_question(self, state: GraphState) -> str:
        """
        Route question to web search or RAG

        Args:
            state (GraphState): The current graph state

        Returns:
            str: Next node to call
        """
        messages = [SystemMessage(ROUTER_INSTRUCTIONS)] + [HumanMessage(state["question"])]
        reasoning = self._llm.invoke(messages)
        messages.extend([reasoning, HumanMessage("Choose the datasource.")])
        route_question = self._llm_json_mode.invoke(messages)
        source = json.loads(route_question.content)["datasource"]
        if source == "websearch":
            return "websearch"
        elif source == "vectorstore":
            return "vectorstore"

    def decide_to_generate(self, state: GraphState) -> Literal["generate", "websearch"]:
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (GraphState): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        web_search = state["web_search"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            return "generate"

    def _construct_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("websearch", self.web_search, input=GraphState)
        builder.add_node("retrieve", self.retrieve, input=GraphState)
        builder.add_node("grade_documents", self.grade_documents, input=GraphState)
        builder.add_node("generate", self.generate, input=GraphState)

        builder.add_conditional_edges(
            START,
            self.route_question,
            path_map={"websearch": "websearch", "vectorstore": "retrieve"},
        )
        builder.add_edge("websearch", "generate")
        builder.add_edge("retrieve", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            path_map={"generate": "generate", "websearch": "websearch"},
        )
        builder.add_edge("generate", END)
        self._graph = builder.compile()


# Post-processing
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
