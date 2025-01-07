from typing import TypedDict, Protocol
import json
import dataclasses
import os

from IPython.display import Image
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig

from custom_docs import CustomDocManager
from search import SearchManager


DOC_GRADER_PROMPT = """

You are a grader assessing the relevance of a retrieved document to a user question.

Here is the retrieved document:
{document}

Carefully and objectively assess whether the document is relevant. Follow these rules.
- If the document addresses some of the concepts in the question, it is relevant.
- If the document contains the keyword(s)from the question, it is relevant.
- If the document contains information, t hat might guide me to find the information on my own, it is relevant.
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

ANSWER_FORM = """
You are an expert writer, that is capable of carefull planning for his next work.

I need you to think about, how the answer to the following question should look like.
The question is:
{question}

Your first thought will include
- the language of the answer - the language is always the same as the question.
- the complexity of the answer - think about, what various information should be included. If the question asks for
a specific fact, the answer should be simple. If the question asks for a complex explanation, a story or research,
the answer should be more complex and include large amount of relevant information.
- the form of the answer - think about, how the answer should be structured. Should it be a list, a paragraph, a table, etc.

Write to me your thoughts as a plain text. Do not write anything else. Do not attempt to answer the question.
"""


PLANNER_INSTRUCTIONS = """
You are an expert in research planning and providing a simple and clear plan for how to get a correct answer to my questions

You are provided with the following information sources with descriptions.
{sources_with_descriptions_dict}

When thinking about the plan, pay attention to these Instructions:
{answer_instructions}

provide a clear plan for how to obtain enough reliable and relevant information to answer the question. Think about the Instructions. If necessary, break down the question into sub-questions, for example:
Question: What are the typical weather conditions on Madeira island?

Sub-question 1: What is the average temperature on Maderia island in each time of the year?
Sub-question 2: Is it rainy on Madeira?

Use these sub-questions as points in your plan.

You will then return a JSON.
Each key is the original question or sub-questions (if the original question is too complex) that will be passed to an information provider, such as a search engine or a database.
The value is a non-empty list of information sources (strings) that you plan to use to answer the question, sorted by priority. If available, try to add search-like tool at the end of the list, if it is not in the list already.
"""


RAG_PROMPT = """
You are an assistant for question-answering tasks.
Here is the context to use to answer the question:
{context}

Provide an answer to this questions using only the above context.

Think carefully about the above context. Keep the anwer language, form and complexity according to the following:
{answer_instructions}

You will then write to me the answer. Do not write anything else.
"""


DocumentID = str
Query = str


@dataclasses.dataclass(frozen=True)
class DocsFromSource:
    documents: list[Document]
    source_name: str


empty_docs = DocsFromSource([], "")


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to and modify in each graph node.
    """

    plan: dict[Query, list[str]]  # Plan for research with sub-questions and sources
    question: str  # User question
    to_be_answered: dict[Query, DocsFromSource]  # List of sub-questions to be answered
    answered: dict[Query, list[Document]]  # Dict of sub-questions with relevant documents
    thoughts_on_answer: str  # Thoughts on the complexity, language and form of the answer
    answer: str  # Answer generated
    max_retries: int  # Max number of retries for answer generation


class Source(Protocol):
    def invoke(self, query: str) -> list[Document]: ...

    @property
    def name(self) -> str: ...


@dataclasses.dataclass(frozen=True)
class DescribedSource:
    source: Source
    description: str

    def invoke(self, query: str) -> list[Document]:
        return self.source.invoke(query)

    @property
    def name(self) -> str:
        return self.source.name


WEB_SEARCH_NAME = "search"


class Retriever:

    def __init__(self, db_root: str) -> None:
        self._document_manager = CustomDocManager()
        self._llm = ChatOpenAI(model="gpt-4o-mini")
        self._grader_llm = ChatOpenAI(model="gpt-3.5-turbo")
        self._grader_llm_json = self._grader_llm.bind(response_format={"type": "json_object"})
        self._llm_json_mode = self._llm.bind(response_format={"type": "json_object"})
        self._construct_graph()
        self._sources: dict[str, DescribedSource] = {
            "documents": DescribedSource(
                source=self._document_manager,
                description="Memory contains verified documents and non-changing information.",
            ),
            "medicine": DescribedSource(
                source=SearchManager("MedicalSearch", db_root, search_type="medical"),
                description=(
                    "Medicine-related topics. PubMed® comprises millions of citations for biomedical literature from MEDLINE, "
                    "life science journals, and online books."
                ),
            ),
            "science": DescribedSource(
                source=SearchManager("ScientificSearch", db_root, search_type="scientific"),
                description=(
                    "Science-related topics and information in 2 million scholarly articles in the fields of physics,"
                    " mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering"
                    " and systems science, and economics."
                ),
            ),
            "langchain_api": DescribedSource(
                source=SearchManager("LangchainAPISearch", db_root, search_type="langchain_api"),
                description="Langchain API search for information.",
            ),
            WEB_SEARCH_NAME: DescribedSource(
                source=SearchManager("WebSearch", db_root, search_type="generic"),
                description="Generic web search for current events or information.",
            ),
        }
        if WEB_SEARCH_NAME not in self._sources:
            raise ValueError(f"Source '{WEB_SEARCH_NAME}' not found in sources. Please add it.")

    @staticmethod
    def cited_answer(state: GraphState) -> str:
        """Return the answer and the sources."""
        docs = [doc for subqdocs in state["answered"].values() for doc in subqdocs]
        docs_sources = dict.fromkeys(str(doc.metadata) for doc in docs).keys()
        return state["answer"] + "\n\n" + "\n\n".join([source for source in docs_sources])

    def add_web_documents(self, *urls: str) -> None:
        self._document_manager.add_web_documents(*urls)

    def add_local_documents(self, *paths: str) -> None:
        self._document_manager.add_local_documents(*paths)

    def answer(self, state: GraphState) -> dict | GraphState:
        """Determine if we need to run web search or generate answer"""
        sub = list(state["to_be_answered"].keys())[0]
        source_names = state["plan"][sub]
        if not isinstance(source_names, list):
            source_names = [source_names]
        print(f"Answering question {sub}. Available sources are: {source_names}.")
        if source_names:
            source_name = source_names.pop(0)
        else:
            source_name = WEB_SEARCH_NAME
        docs = self._get_docs_from_source(source_name, sub)
        state["to_be_answered"].update({sub: DocsFromSource(docs, source_name)})
        return state

    def invoke(self, query: str, debug: bool = False) -> GraphState:
        config = RunnableConfig(recursion_limit=50)
        return self._graph.invoke(
            GraphState(
                question=query,
                to_be_answered=dict(),
                plan=dict(),
                answer="",
                answered={},
                thoughts_on_answer="",
                max_retries=5,
            ),
            debug=debug,
            config=config,
        )

    def generate(self, state: GraphState) -> dict:
        """Generate answer using RAG on retrieved documents."""
        question: str = state["question"]

        # RAG generation
        docs_txt = format_docs(state["answered"])
        rag_prompt_formatted = RAG_PROMPT.format(
            context=docs_txt, question=question, answer_instructions=state["thoughts_on_answer"]
        )
        result = self._llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"answer": result.content}

    def grade_documents(self, state: GraphState) -> dict | GraphState:
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search
        """
        question = list(state["to_be_answered"].keys())[0]
        docs_from_source = state["to_be_answered"][question]
        documents = docs_from_source.documents
        # Score each doc
        filtered_docs: list[Document] = []
        for d in documents:
            doc_grader_prompt_formatted = DOC_GRADER_PROMPT.format(
                document=d.page_content, question=question
            )
            messages = [SystemMessage(doc_grader_prompt_formatted), HumanMessage(question)]
            thought = self._grader_llm.invoke(messages)
            messages.extend(
                [thought, HumanMessage("Grade the document relevance with 'yes' or 'no'.")]
            )
            result = self._grader_llm_json.invoke(messages)
            grade = str(json.loads(str(result.content))["binary_score"])
            if grade.lower() == "yes":  # Document relevant
                filtered_docs.append(d)

        answered = state.get("answered", {})
        sources = state["plan"][question]
        if len(filtered_docs) > 1 or not sources:
            state["to_be_answered"].pop(question)
            answered[question] = filtered_docs
        state["answered"] = answered
        return state

    def think_about_answer(self, state: GraphState) -> dict:
        """Think about the complexity, language and form of the answer"""
        question = state["question"]
        thought = self._llm.invoke([SystemMessage(ANSWER_FORM.format(question=question))])
        print("Thoughts on the answer:\n", str(thought.content))
        return {"thoughts_on_answer": thought.content}

    def plan(self, state: GraphState) -> dict:
        """Plan the research based on the user question and available sources"""
        sources = {k: v.description for k, v in self._sources.items()}
        messages = [
            SystemMessage(
                PLANNER_INSTRUCTIONS.format(
                    sources_with_descriptions_dict=sources,
                    answer_instructions=state["thoughts_on_answer"],
                )
            ),
            HumanMessage(state["question"]),
        ]
        plan: dict[str, str] = json.loads(str(self._llm_json_mode.invoke(messages).content))
        print("The plan for getting the necessary information:\n", json.dumps(plan, indent=4))
        return {
            "plan": plan,
            "to_be_answered": {key: empty_docs for key in plan.keys()},
        }

    def _get_docs_from_source(self, source_name: str, query: Query) -> list[Document]:
        try:
            print(f"Source '{source_name}' will be used to answer '{query}'.")
            docs = self._sources[source_name].invoke(query)
        except Exception as e:
            print(f"Error when invoking source '{source_name}': {str(e)}")
            docs = []
        return docs

    def check_questions(self, state: GraphState) -> str:
        """Determine if we need to run web search or generate answer"""
        return "next" if bool(state["to_be_answered"]) else "generate"

    def _construct_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("think_about_answer", self.think_about_answer, input=GraphState)
        builder.add_node("create_plan", self.plan, input=GraphState)
        builder.add_node("answer_next", self.answer, input=GraphState)
        builder.add_node("grade_documents", self.grade_documents, input=GraphState)
        builder.add_node("generate", self.generate, input=GraphState)

        builder.add_edge(START, "think_about_answer")
        builder.add_edge("think_about_answer", "create_plan")
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
        return output_state["answer"] + "\n\n" + "\n\n".join([str(doc.metadata) for doc in docs])

    def print_graph_png(self, path: str, name: str = "retriever") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)


# Post-processing
def format_docs(docs_dict: dict[str, list[Document]]) -> str:
    return "\n\n".join(
        subq + "\n" + "-" * len(subq) + "\n".join(doc.page_content for doc in docs)
        for subq, docs in docs_dict.items()
    )
