from graph import RetrieverWithPlan


retriever = RetrieverWithPlan()


retriever.print_graph_png(".")
# retriever.add_web_documents("http://www.planivy.cz/index.php?page=planivy&section=jeskyne")


result = retriever.invoke("Kdy byla objevena Půlnoční propast v závrtu Okrouhlík?")
print(result)
