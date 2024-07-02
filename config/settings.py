import os
from dotenv import load_dotenv
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-3.5-turbo")

Settings.llm = llm
Settings.embed_model = embed_model
