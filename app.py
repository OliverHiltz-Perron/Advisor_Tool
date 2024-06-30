import os
import pandas as pd
import openai
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage, Document, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import json
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-3.5-turbo")

Settings.llm = llm
Settings.embed_model = embed_model

# Prompt Templates
def load_template(file_name):
    with open(file_name, 'r') as file:
        return file.read()

# Load templates from files
router_prompt = load_template('Prompts/router_prompt.txt')
templates = {
    'student': PromptTemplate(load_template('Prompts/text_qa_template_student.txt')),
    'prof': PromptTemplate(load_template('Prompts/text_qa_template_prof.txt')),
    'courses': PromptTemplate(load_template('Prompts/text_qa_template_courses.txt'))
}

#Models the ai can choose from
choices = [
    "Question relates to the course descriptions",
    "Questions relate to the faculty profiles",
    "Question relates to the student guide"
]

def get_choice_str(choices):
    return "\n\n".join([f"{idx+1}. {c}" for idx, c in enumerate(choices)])


choices_str = get_choice_str(choices)

#sets up the models
def initialize_vector_store(db_path, collection_name):
    db_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = db_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

index_classes = initialize_vector_store("./chroma_db_classes", "AdvisorTool_classes")
index = initialize_vector_store("./chroma_db", "AdvisorTool")
index_prof = initialize_vector_store("./chroma_db_professor", "AdvisorTool_Professor")

####################################################
#  User-defined functions for query and response   #
####################################################


def get_formatted_prompt(query_str, chat_history_str):
    fmt_prompt = router_prompt.format(
        num_choices=3,
        max_outputs=2,
        context_list=choices_str,
        query_str=query_str,
        chat_history=chat_history_str  # Include chat history in the prompt
    )
    response = llm.complete(fmt_prompt)
    return response.text

def format_chat_history(chat_history):
    history_str = ""
    for idx, (query, response, context) in enumerate(chat_history):
        history_str += f"**Query {idx + 1}:** {query}\n\n"
        history_str += f"**Response {idx + 1}:** {response}\n\n"
        history_str += f"**Context {idx + 1}:** {context}\n\n"
    return history_str



# Query index and return results
def query_index(query, index, template, chat_history_str):
    full_query = template.format(
        query_str=query,
        chat_history=chat_history_str,
        context_str="{context_str}"  # Placeholder to be filled by the query engine
    )
    response = index.as_query_engine(
        text_qa_template=PromptTemplate(full_query),
        similarity_top_k=8,
    ).query(query)
    return response

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
import re

def extract_choice_index(response):
    match = re.search(r'\d+', response)
    return int(match.group()) if match else None

def perform_query(query):
    chat_history_str = format_chat_history(chat_history)
    prompt_response = get_formatted_prompt(query, chat_history_str)
    choice_index = extract_choice_index(prompt_response)

    if choice_index == 1:
        response_data = query_index(query, index, templates['student'], chat_history_str)
    elif choice_index == 2:
        response_data = query_index(query, index_prof, templates['prof'], chat_history_str)
    elif choice_index == 3:
        response_data = query_index(query, index_classes, templates['courses'], chat_history_str)
    else:
        response_data = "I'm sorry, I couldn't determine the context of your question. Please try again."

    response_str, context = process_response(response_data)
    return response_str, context


def answer_question(question):
    response, context = perform_query(question)
    response_md = f"**Response:**\n\n{response}"
    context_md = f"**Context:**\n\n{context}"
    return gr.Markdown(response_md), gr.Markdown(context_md)

#######################
#  Gradio front end   #
#######################

import gradio as gr

# Global variables to store response and context
response_global = ""
context_global = ""

# Initialize chat history
chat_history = []

# Function to get response and update chat history
def get_response(query):
    response, context = perform_query(query)
    chat_history.append((query, response, context))
    return response, context


def show_response(query):
    response, _ = get_response(query)
    return response

def show_context(query):
    _, context = get_response(query)
    return context

def display_chat_history():
    history_md = ""
    for idx, (query, response, context) in enumerate(chat_history):
        history_md += f"**Query {idx + 1}:** {query}\n\n"
        history_md += f"**Response {idx + 1}:** {response}\n\n"
        history_md += f"**Context {idx + 1}:** {context}\n\n"
    return gr.Markdown(history_md)

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