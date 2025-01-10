import sys
import os

sys.path.append(".")

from search import SearchManager


def main():
    storage_path = os.path.join(os.path.dirname(__file__), "memory")
    search = SearchManager("search", storage_path=storage_path)
    query = "What are the perspectives of the RAG systems and what are their disadvantages? What is their application in economics?"
    response = search.invoke(query)


if __name__ == "__main__":
    main()
