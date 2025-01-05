import os

from graph import Retriever


# Initialize the retriever with db_root pointing to some (even nonexistent) directory
retriever = Retriever(db_root=os.path.dirname(__file__) + "/.memory")
write_to = os.path.join(os.path.dirname(__file__), "answers.txt")


query = ""
# main loop. Type 'exit' to quit. Type any query in any language to get the answer in the same language
while query != "exit":
    query = input("Enter query or 'exit': ")
    print("Your query:", query)
    result = retriever.invoke(query)
    print("-" * 80)
    print("Answer:\n")
    print(Retriever.cited_answer(result))

    try:
        with open(write_to, "a") as f:
            f.write("=" * 80)
            f.write("Input: " + query)
            f.write("\n")
            f.write("-" * 80)
            f.write("Answer:\n")
            f.write(Retriever.cited_answer(result))
    except:
        print(f"Failed to write to file {write_to}")
