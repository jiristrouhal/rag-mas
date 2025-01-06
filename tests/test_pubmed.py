import sys

sys.path.append(".")

from search import PubMedSearch


search = PubMedSearch("Pubmed")


docs = search.invoke("severe forms of cancer")
print(docs)
