import os
import pandas as pd
import openai
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage, Document, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import json
import re
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
text_qa_template_student = PromptTemplate(load_template('Prompts/text_qa_template_student.txt'))
text_qa_template_prof = PromptTemplate(load_template('Prompts/text_qa_template_prof.txt'))
text_qa_template_courses = PromptTemplate(load_template('Prompts/text_qa_template_courses.txt'))

choices_str = load_template('Prompts/choices.txt')

db2 = chromadb.PersistentClient(path="./chroma_db_classes")
chroma_collection = db2.get_or_create_collection("AdvisorTool_Courses")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index_classes = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("AdvisorTool")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)
# load from disk
db2p = chromadb.PersistentClient(path="./chroma_db_professor")
chroma_collection = db2p.get_or_create_collection("AdvisorTool_Professor")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index_pro = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)


####################################################
#  User-defined functions for query and response   #
####################################################


def get_formatted_prompt(query_str):
    fmt_prompt = router_prompt.format(
        num_choices=3,
        max_outputs=2,
        context_list=choices_str,
        query_str=query_str,
    )
    response = llm.complete(fmt_prompt)
    return response.text

# Query index and return results
def query_index_student(query):
    response = index.as_query_engine(
        text_qa_template=text_qa_template_student,
        similarity_top_k=8,
    ).query(query)
    return response

def query_index_prof(query):
    response = index_pro.as_query_engine(
        text_qa_template=text_qa_template_prof,
        similarity_top_k=8,
    ).query(query)
    return response

def query_index_classes(query):
    response = index_classes.as_query_engine(
        text_qa_template=text_qa_template_courses,
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


def extract_choice_index(response):
    match = re.search(r'\d+', response)
    if match:
        return int(match.group())
    else:
        return None

def perform_query(query):
    prompt_response = get_formatted_prompt(query)
    choice_index = extract_choice_index(prompt_response)

    if choice_index == 1:
        response_data = query_index_student(query)
    elif choice_index == 2:
        response_data = query_index_prof(query)
    elif choice_index == 3:
        response_data = query_index_classes(query)
    else:
        response_data = "I'm sorry, I couldn't determine the context of your question. Please try again."
    # Modify the process_response function to return the response as a string
    response_str = process_response(response_data)

    return response_str

def answer_question(question):
    response, context = perform_query(question)
    response_md = f"**Response:**\n\n{response}"
    context_md = f"**Context:**\n\n{context}"
    return gr.Markdown(response_md), gr.Markdown(context_md)

import gradio as gr

# Global variables to store response and context
response_global = ""
context_global = ""

def get_response(query):
    prompt_response = get_formatted_prompt(query)
    choice_index = extract_choice_index(prompt_response)

    if choice_index == 1:
        response_data = query_index_student(query)
    elif choice_index == 2:
        response_data = query_index_prof(query)
    elif choice_index == 3:
        response_data = query_index_classes(query)
    else:
        response_data = "I'm sorry, I couldn't determine the context of your question. Please try again."

    response, context = process_response(response_data)
    return response, context

def show_response(query):
    response, _ = get_response(query)
    return response

def show_context(query):
    _, context = get_response(query)
    return context

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Academic Advisor Assistant
        
        Welcome to the Academic Advisor Assistant! This AI-powered tool can help answer questions about:
        - Course information and descriptions
        - Faculty profiles and research interests
        - Student handbook policies and requirements
        
        Simply type your question below and click 'Submit' to get started.
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What courses are available for the Fall semester?",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")
                
        with gr.Tabs():
            with gr.TabItem("Answer"):
                response_output = gr.Markdown(label="Response")
            with gr.TabItem("Source Details"):
                context_output = gr.Markdown(label="Context")
        
        # Set up event handlers
        submit_btn.click(
            fn=get_response,
            inputs=query_input,
            outputs=[response_output, context_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=[],
            outputs=[query_input, response_output, context_output]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                "What courses are required for graduation?",
                "Who teaches artificial intelligence?",
                "What are the prerequisites for advanced courses?",
            ],
            inputs=query_input,
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)