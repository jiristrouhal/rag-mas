import sys

sys.path.append(".")

from search import PubMedSearch


search = PubMedSearch("Pubmed")


def main():
    docs = search.invoke("severe forms of cancer")
    print(docs)


if __name__ == "__main__":
    main()
