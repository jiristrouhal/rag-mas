from graph import Retriever, GraphState


retriever = Retriever()
retriever.add_web_documents("https://www.speleo.cz/amaterska-jeskyne")
retriever.add_web_documents("https://www.speleo.cz")

result = retriever._graph.invoke(GraphState(question="What is the speed of light?"))

print(result)
