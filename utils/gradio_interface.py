import gradio as gr
from utils.query_processing import get_formatted_prompt, extract_choice_index, query_index, format_chat_history, get_choice_str
from utils.response_processing import process_response
from config.settings import embed_model, llm
from utils.template_loader import load_templates
from utils.vector_store import initialize_vector_store

chat_history = []

choices = [
    "Question relates to the course descriptions",
    "Questions relate to the faculty profiles",
    "Question relates to the student guide"
]
choices_str = get_choice_str(choices)

def perform_query(query):
    router_prompt, templates = load_templates()
    index_classes = initialize_vector_store("./data/chroma_db_classes", "AdvisorTool_classes", embed_model)
    index = initialize_vector_store("./data/chroma_db", "AdvisorTool", embed_model)
    index_prof = initialize_vector_store("./data/chroma_db_professor", "AdvisorTool_Professor", embed_model)

    chat_history_str = format_chat_history(chat_history)
    prompt_response = get_formatted_prompt(router_prompt, query, chat_history_str, choices_str, llm)
    choice_index = extract_choice_index(prompt_response)

    if choice_index == 1:
        template = templates['student'].format(query_str=query, chat_history=chat_history_str, context_str="{context_str}")
        response_data = query_index(query, index, template, chat_history_str)
    elif choice_index == 2:
        template = templates['prof'].format(query_str=query, chat_history=chat_history_str, context_str="{context_str}")
        response_data = query_index(query, index_prof, template, chat_history_str)
    elif choice_index == 3:
        template = templates['courses'].format(query_str=query, chat_history=chat_history_str, context_str="{context_str}")
        response_data = query_index(query, index_classes, template, chat_history_str)
    else:
        response_data = "I'm sorry, I couldn't determine the context of your question. Please try again."

    response_str, context = process_response(response_data)
    return response_str, context

def show_response(query):
    response, _ = perform_query(query)
    return response

def show_context(query):
    _, context = perform_query(query)
    return context

def display_chat_history():
    history_md = ""
    for idx, (query, response, context) in enumerate(chat_history):
        history_md += f"**Query {idx + 1}:** {query}\n\n"
        history_md += f"**Response {idx + 1}:** {response}\n\n"
        history_md += f"**Context {idx + 1}:** {context}\n\n"
    return gr.Markdown(history_md)

def launch_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Advisor Support App\n\nThis app draws on information from course descriptions, faculty profiles, and the student handbook. Questions that are outside of this scope cannot be answered by the app.")

        with gr.Tabs():
            with gr.TabItem("Response"):
                query_input = gr.Textbox(lines=2, placeholder="Enter your query here...")
                response_output = gr.Markdown(label="Response")
                query_button = gr.Button("Submit")
                clear_button = gr.Button("Clear")
                query_button.click(fn=show_response, inputs=query_input, outputs=response_output)
                clear_button.click(lambda: "", None, query_input)
                clear_button.click(lambda: "", None, response_output)
            with gr.TabItem("Context"):
                context_output = gr.Markdown(label="Context")
                query_input_context = gr.Textbox(lines=2, placeholder="Enter your query here...", visible=False)
                query_button.click(fn=show_context, inputs=query_input, outputs=context_output)
                clear_button.click(lambda: "", None, context_output)

    demo.launch(share=True)

# Ensure the function is defined properly
