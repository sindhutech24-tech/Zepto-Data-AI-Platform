from main import graph

policy = graph.invoke({"query": "What is the delivery fee below INR 149?"})
general = graph.invoke({"query": "What is the capital of India?"})

print("POLICY:")
print(policy["answer"].model_dump_json())
print("\nGENERAL:")
print(general["answer"].model_dump_json())
