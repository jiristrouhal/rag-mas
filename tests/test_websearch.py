import sys

sys.path.append(".")

from search import WebSearch


search = WebSearch("WebSearch")


docs = search.invoke("severe forms of cancer")
print(docs)
