import sys
import os

sys.path.append(".")


from search import SearchManager


def main():
    storage_path = os.path.join(os.path.dirname(__file__), "memory")
    search = SearchManager(storage_path=storage_path, use_concept=True)

    query = "závrt Okrouhlík?"
    response = search.invoke(query)
    for r in response:
        print(r)


if __name__ == "__main__":
    main()
