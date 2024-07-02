def process_response(response_data):
    if isinstance(response_data, str):
        return "Response: " + response_data
    else:
        response = "Response: " + response_data.response + "\n"
        context = ""
        for node in response_data.source_nodes:
            context += "\nRelevance Score: " + f"{node.score:.2f}\n"
            for key, value in node.node.metadata.items():
                context += f"{key}: {value}\n"
            context += "Text: " + node.node.text + "\n"
        return response, context
