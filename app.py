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


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-3.5-turbo")

Settings.llm = llm
Settings.embed_model = embed_model

# Prompt Templates

router_prompt = PromptTemplate("""
    Some choices are given below. It is provided in a numbered list (1 to {num_choices}), where each item in the list corresponds to a
    summary. If you have any doubt, select the student guide.\n---------------------\n{context_list}\n---------------------\n
    
    Use this information to help meake a determination:
    - Course descriptions: These questions are be about course content, course objectives, course descriptions, and topics. Course
      descriptions do not have any information about faculty, instructors, field education, or graduation requirements. \n
    
    - Faculty profiles: These questions will be about faculty members, their research, their teaching interests. Any question about 
    person will be answered here. Any question about "who" studies a given topic will be answered by the faculty profiles \n
    
    -Student guide: Any questions that are about credits, internships, enrollment, graduation, policies, and other student requirements fit this category.
    
    
    Using only the information provide, return the top choices (no more than {max_outputs}, but only select what is needed) 
    that are most relevant to the question: '{query_str}'\n
""")


choices = [
    "Question relates to the course descriptions",
    "Questions relate to the faculty profiles",
    "Question relates to the student guide"
]

text_qa_template_student_str = (
    '''Context information about the student guide or handbook is
        \n---------------------\n{context_str}\n---------------------\n
    
    Using the context information, answer the question: {query_str}
    
    Provide a detailed response in a friendly voice. Do not say anything irrelevant. Please note the following:
    - A field placement may be called an internship, field, practicum or placement.\n
    
    If you cannot answer the question based on the context provided, tell the user that you don't have access to that specific information. \n
    When responding, refer to the "context" as your "available information."
    For unanswered questions, direct them to the technical advising team at ssw.technicaladvising@umich.edu. 
  
    \n
    \n

  '''
)

text_qa_template_prof_str = (
    '''Context information about faculty members is below.
    \n---------------------\n{context_str}\n---------------------\n
    
    Using the context information, answer the question: {query_str}
    
    Follow these rules when responding:
    - Provide a detailed response in a friendly voice. 
    - Interpret the question broadly, trying to find faculty that are relevant. 
    - Summarize only the faculty who are directly relevant. Do not provide any irrelevant information.
    - If you cannot answer the question based on the context provided, tell the user that you don't have access to that specific information. \n
    - When responding, refer to the "context" as your "available information."
    '''
)

text_qa_template_courses_str = (
    
)


text_qa_template_student = PromptTemplate(text_qa_template_student_str)
text_qa_template_prof = PromptTemplate(text_qa_template_prof_str)
text_qa_template_courses = PromptTemplate(text_qa_template_courses_str)



def get_choice_str(choices):
    choices_str = "\n\n".join(
        [f"{idx+1}. {c}" for idx, c in enumerate(choices)]
    )
    return choices_str


choices_str = get_choice_str(choices)


# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db_classes")
chroma_collection = db2.get_or_create_collection("AdvisorTool_classes")
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
        # num_choices=len(choices),
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

import re

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
    global response_global, context_global
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

    response_global, context_global = process_response(response_data)
    return response_global, context_global

def show_response(query):
    response, _ = get_response(query)
    return response

def show_context(query):
    _, context = get_response(query)
    return context

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