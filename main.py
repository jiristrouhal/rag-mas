from graph import Retriever


retriever = Retriever()


query = ""
while query != "exit":
    query = input("Enter query or 'exit': ")
    print("Your query:", query)
    result = retriever.invoke(query)
    print("-" * 80)
    print("Answer:\n")
    print(Retriever.cited_answer(result))
