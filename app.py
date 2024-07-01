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

# Load environment variables from a .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client and models
client = OpenAI(api_key=api_key)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-3.5-turbo")

# Set global settings for LLM and embedding model
Settings.llm = llm
Settings.embed_model = embed_model

# Function to load a template from a file
def load_template(file_name):
    with open(file_name, 'r') as file:
        return file.read()

# Load prompt templates from files
router_prompt = load_template('Prompts/router_prompt.txt')
templates = {
    'student': PromptTemplate(load_template('Prompts/text_qa_template_student.txt')),
    'prof': PromptTemplate(load_template('Prompts/text_qa_template_prof.txt')),
    'courses': PromptTemplate(load_template('Prompts/text_qa_template_courses.txt'))
}

# Define choices for the AI to choose from
choices = [
    "Question relates to the course descriptions",
    "Questions relate to the faculty profiles",
    "Question relates to the student guide"
]

# Function to format choices as a string
def get_choice_str(choices):
    return "\n\n".join([f"{idx+1}. {c}" for idx, c in enumerate(choices)])

# Format choices string
choices_str = get_choice_str(choices)

# Function to initialize a vector store
def initialize_vector_store(db_path, collection_name):
    db_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = db_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Initialize vector stores for different collections
index_classes = initialize_vector_store("./chroma_db_classes", "AdvisorTool_classes")
index = initialize_vector_store("./chroma_db", "AdvisorTool")
index_prof = initialize_vector_store("./chroma_db_professor", "AdvisorTool_Professor")

####################################################
#  User-defined functions for query and response   #
####################################################

# Function to format the prompt with the query and chat history
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

# Function to format chat history into a string
def format_chat_history(chat_history):
    history_str = ""
    for idx, (query, response, context) in enumerate(chat_history):
        history_str += f"**Query {idx + 1}:** {query}\n\n"
        history_str += f"**Response {idx + 1}:** {response}\n\n"
    return history_str

# Function to query the index and return results
def query_index(query, index, template, chat_history_str):
    full_query = template  # The template already includes the chat history and query string
    if index is index_prof:
        response = index.as_query_engine(
            text_qa_template=PromptTemplate(full_query),
            similarity_top_k=5,
        ).query(query)
        return response
    else:
        response = index.as_query_engine(
            text_qa_template=PromptTemplate(full_query),
            similarity_top_k=4,
        ).query(query)
        return response

# Function to process the response and format it
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

# Function to extract choice index from the response
def extract_choice_index(response):
    match = re.search(r'\d+', response)
    return int(match.group()) if match else None

# Main function to perform a query and get the response
def perform_query(query):
    chat_history_str = format_chat_history(chat_history)
    prompt_response = get_formatted_prompt(query, chat_history_str)
    choice_index = extract_choice_index(prompt_response)
    
    if choice_index == 1:
        template = templates['student'].template.format(
            query_str=query,
            chat_history=chat_history_str,
            context_str="{context_str}"
        )
        response_data = query_index(query, index, template, chat_history_str)
    elif choice_index == 2:
        template = templates['prof'].template.format(
            query_str=query,
            chat_history=chat_history_str,
            context_str="{context_str}"
        )
        response_data = query_index(query, index_prof, template, chat_history_str)
    elif choice_index == 3:
        template = templates['courses'].template.format(
            query_str=query,
            chat_history=chat_history_str,
            context_str="{context_str}"
        )
        response_data = query_index(query, index_classes, template, chat_history_str)
    else:
        response_data = "I'm sorry, I couldn't determine the context of your question. Please try again."

    response_str, context = process_response(response_data)
    return response_str, context

# Function to answer a question and format the response
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

# Function to show response
def show_response(query):
    response, _ = get_response(query)
    return response

# Function to show context
def show_context(query):
    _, context = get_response(query)
    return context

# Function to display chat history
def display_chat_history():
    history_md = ""
    for idx, (query, response, context) in enumerate(chat_history):
        history_md += f"**Query {idx + 1}:** {query}\n\n"
        history_md += f"**Response {idx + 1}:** {response}\n\n"
        history_md += f"**Context {idx + 1}:** {context}\n\n"
    return gr.Markdown(history_md)

# Define the Gradio interface
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

# Launch the Gradio app
demo.launch(share=True)
