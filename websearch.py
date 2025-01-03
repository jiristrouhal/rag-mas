import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


dotenv.load_dotenv()


PROMPT = """
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


class WebSearcher:
    """Search the web for information related to the question."""

    def __init__(self):
        self._web_search_tool = TavilySearchResults(max_results=2)
        self._translator = ChatOpenAI(model="gpt-3.5-turbo")

    def invoke(self, question: str) -> list[Document]:
        """Search the web for information related to the question.

        Args:
            question (str): The question to search for.
        Returns:
            list[Document]: The documents found on the web.
        """
        docs: list = self._web_search_tool.invoke({"query": question})
        translated_concept = self.translate_concept(question)
        if translated_concept:
            docs = self._web_search_tool.invoke({"query": translated_concept}) + docs
        documents = []
        for d in docs:
            web_result = "\n".join([d["content"] for d in docs])
            web_result = Document(page_content=web_result, metadata={"source": d["url"]})
            documents.append(web_result)
        return documents

    def translate_concept(self, question: str) -> str:
        """Translate the key concept from the question to the best language to search for information.

        For example:
        Question: What is the capital of Portugal?
        Translated concept: capitál de Portugal

        Args:
            question (str): The question to translate.
        Returns:
            str: The translated concept
        """
        return self._translator.invoke([SystemMessage(PROMPT), HumanMessage(question)]).content
