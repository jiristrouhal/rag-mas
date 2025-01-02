from graph import RetrieverSimple


retriever = RetrieverSimple()


retriever.print_graph_png(".")
# retriever.add_web_documents("http://www.planivy.cz/index.php?page=planivy&section=jeskyne")


result = retriever.invoke("What is the date of discovery of the Stará Amatérská jeskyně?")
print(result)
