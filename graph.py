import operator
from typing import TypedDict, Annotated, Protocol
import json
import os

from IPython.display import Image
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from memory import Memory
from websearch import WebSearcher


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

PLANNER_INSTRUCTIONS = """
You are an expert in research planning and providing a simple and clear plan for how to get a correct answer to my questions

You are provided with the following information sources with descriptions.
{sources_with_descriptions_dict}

Based on the information sources and the user's question, provide a simple and clear plan for how to obtain enough reliable and relevant information to answer the question.

Think about the complexity of the question and if necessary, break it down into more sub-questions.

You will then return a JSON.
Each key is the original question or sub-questions (if the original question is too complex) that will be passed to an information provider, such as a search engine or a database.
The value is a non-empty list of information sources (strings) that you plan to use to answer the question, sorted by priority. If available, try to add search-like tool at the end of the list, if it is not in the list already.
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

Respond in the same language as the question.

Answer:
"""


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to and modify in each graph node.
    """

    plan: dict[str, list[str]]  # Plan for research with sub-questions and sources
    question: str  # User question
    to_be_answered: dict[str, list[Document]]  # List of sub-questions to be answered
    answered: dict[str, list[Document]]  # Dict of sub-questions with relevant documents
    answer: str  # Answer generated
    max_retries: int  # Max number of retries for answer generation
    loop_step: Annotated[int, operator.add]

    def cited_answer(self) -> str:
        """Return the answer and the sources."""
        docs = [doc for subqdocs in self["answered"].values() for doc in subqdocs]
        return self["answer"] + "\n\n" + "\n\n".join([doc.metadata["source"] for doc in docs])


class CitedAnswer(TypedDict):
    """Cited answer is a dictionary that contains information about the answer and its source."""

    answer: str  # Answer generated
    sources: list[Document]  # List of sources for the answer

    def __str__(self) -> str:
        return self["answer"] + "\n\n" + "\n".join(self["sources"])

    @staticmethod
    def cited_document(doc: Document) -> str:
        """Return the metadata of the document."""
        return f"{doc.metadata}"


class Source(Protocol):
    def invoke(self, state: GraphState) -> dict: ...


class DescribedSource(TypedDict):
    source: Source
    description: str


class RetrieverWithPlan:

    def __init__(self):
        self._document_manager = Memory()
        self._llm = ChatOpenAI(model="gpt-4o-mini")
        self._llm_json_mode = self._llm.bind(response_format={"type": "json_object"})
        self._websearch_tool = WebSearcher()
        self._construct_graph()
        self._sources: dict[str, DescribedSource] = {
            "memory": DescribedSource(
                source=self._document_manager,
                description="Memory contains verified documents and non-changing information.",
            ),
            "search": DescribedSource(
                source=self._websearch_tool,
                description="Web search for current events or information.",
            ),
        }

    def add_web_documents(self, *urls: str) -> None:
        self._document_manager.add_web_documents(*urls)

    def add_local_documents(self, *paths: str) -> None:
        self._document_manager.add_local_documents(*paths)

    def invoke(self, query: str) -> str:
        return self._graph.invoke(GraphState(question=query))

    def generate(self, state: GraphState) -> dict:
        """Generate answer using RAG on retrieved documents."""
        question: str = state["question"]
        documents: list[Document] = [
            doc for subqdocs in state["answered"].values() for doc in subqdocs
        ]
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
        """
        question = list(state["to_be_answered"].keys())[0]
        documents: list[Document] = state["to_be_answered"][question]
        # Score each doc
        filtered_docs = []
        for d in documents:
            doc_grader_prompt_formatted = DOC_GRADER_PROMPT.format(
                document=d.page_content, question=question
            )
            messages = [
                SystemMessage(content=doc_grader_prompt_formatted),
                HumanMessage(content=question),
            ]
            thought = self._llm.invoke(messages)
            messages.extend(
                [thought, HumanMessage("Grade the document relevance with 'yes' or 'no'.")]
            )
            result = self._llm_json_mode.invoke(messages)
            grade = str(json.loads(result.content)["binary_score"])
            if grade.lower() == "yes":  # Document relevant
                filtered_docs.append(d)

        answered = state.get("answered", {})
        sources = state["plan"][question]
        if len(filtered_docs) > 1 or not sources:
            state["to_be_answered"].pop(question)
            answered[question] = filtered_docs
        state["answered"] = answered
        return state

    def plan(self, state: GraphState) -> dict:
        """Plan the research based on the user question and available sources"""
        sources = {k: v["description"] for k, v in self._sources.items()}
        messages = [
            SystemMessage(PLANNER_INSTRUCTIONS.format(sources_with_descriptions_dict=sources)),
            HumanMessage(state["question"]),
        ]
        plan: dict[str, str] = json.loads(self._llm_json_mode.invoke(messages).content)
        return {"plan": plan, "to_be_answered": {key: [] for key in plan.keys()}}

    def answer(self, state: GraphState) -> dict:
        """Determine if we need to run web search or generate answer"""
        sub = list(state["to_be_answered"].keys())[0]
        if state["plan"][sub]:
            try:
                source: str = state["plan"][sub].pop(0)
                docs = self._sources[source]["source"].invoke(sub)
            except:
                docs = []
        state["to_be_answered"].update({sub: docs})
        return state

    def check_questions(self, state: GraphState) -> str:
        """Determine if we need to run web search or generate answer"""
        return "next" if bool(state["to_be_answered"]) else "generate"

    def _construct_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("create_plan", self.plan, input=GraphState)
        builder.add_node("answer_next", self.answer, input=GraphState)
        builder.add_node("grade_documents", self.grade_documents, input=GraphState)
        builder.add_node("generate", self.generate, input=GraphState)

        builder.add_edge(START, "create_plan")
        builder.add_edge("create_plan", "answer_next")
        builder.add_conditional_edges(
            "grade_documents",
            self.check_questions,
            path_map={"next": "answer_next", "generate": "generate"},
        )
        builder.add_edge("answer_next", "grade_documents")
        builder.add_edge("generate", END)
        self._graph = builder.compile()

    def _format_invoke_response(self, output_state: GraphState) -> str:
        docs = [doc for subqdocs in output_state["answered"].values() for doc in subqdocs]
        return (
            output_state["answer"] + "\n\n" + "\n\n".join([doc.metadata["source"] for doc in docs])
        )

    def print_graph_png(self, path: str, name: str = "retriever") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)


# Post-processing
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
