import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


dotenv.load_dotenv()


web_search_tool = TavilySearchResults(max_results=2)
