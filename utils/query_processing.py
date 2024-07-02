import re
from llama_index.core import PromptTemplate

def get_formatted_prompt(router_prompt, query_str, chat_history_str, choices_str, llm):
    fmt_prompt = router_prompt.format(
        num_choices=3,
        max_outputs=2,
        context_list=choices_str,
        query_str=query_str,
        chat_history=chat_history_str
    )
    response = llm.complete(fmt_prompt)
    return response.text

def extract_choice_index(response):
    match = re.search(r'\d+', response)
    return int(match.group()) if match else None

def query_index(query, index, template, chat_history_str):
    response = index.as_query_engine(
        text_qa_template=PromptTemplate(template),
        similarity_top_k=5  # Adjust similarity_top_k as needed
    ).query(query)
    return response

def format_chat_history(chat_history):
    history_str = ""
    for idx, (query, response, context) in enumerate(chat_history):
        history_str += f"**Query {idx + 1}:** {query}\n\n"
        history_str += f"**Response {idx + 1}:** {response}\n\n"
    return history_str

def get_choice_str(choices):
    return "\n\n".join([f"{idx+1}. {c}" for idx, c in enumerate(choices)])
