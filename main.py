from graph import RetrieverWithPlan


retriever = RetrieverWithPlan()


retriever.print_graph_png(".")
retriever.add_local_documents("test_folder")


result = retriever.invoke("How to make the LLM multi-agent system to learn from experience?")
print(result)
