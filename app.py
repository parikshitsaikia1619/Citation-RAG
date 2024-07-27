import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser,SimpleNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer

import torch
import gradio as gr
import re

# Load tokenizer and llm
tokenizer = AutoTokenizer.from_pretrained(
"meta-llama/Meta-Llama-3-8B-Instruct"
)

stopping_ids = [
tokenizer.eos_token_id,
tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    context_window=4096,
    max_new_tokens=512,
    model_kwargs={'trust_remote_code':True},
    generate_kwargs={"do_sample": False},
    device_map="auto",
    stopping_ids=stopping_ids,
)

embed_model= HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

Settings.embed_model = embed_model
Settings.llm = llm

# Read the pdf
documents = SimpleDirectoryReader("../book/").load_data()

print(f"no. of doc chunks: {len(documents)}\n")

# Create sentence window node parser with default settings
sentence_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
    include_metadata=False,
)

# Parse documents into nodes
sentence_nodes = sentence_node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

query_engine = CitationQueryEngine.from_args(
    sentence_index,
    citation_chunk_size=512,
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
    llm=llm,
)

response = query_engine.query("What is taming intuitive predictions ?")
print(response.response)


# Global variable to store citation_text temporarily
temporary_citation_text = ""

# Function to handle query and return response and citations
def search_query(query):
    global temporary_citation_text
    
    response = query_engine.query(query)
    response_text = response.response.split("\n")[0]

    pattern = r'\[(\d+)\]'
    # Find all matches
    matches = re.findall(pattern, response_text)

    citation_idx = set()
    for match in matches:
        citation_idx.add(int(match)-1)
        
    citation_idx = list(citation_idx)
    print(citation_idx)
    
    citations = []
    metakeys = ['page_label', 'file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date']
    for cit in citation_idx:
        meta_text = ""
        for key in metakeys:
            meta_text+=f"{key}: {response.source_nodes[cit].metadata[key]}\n"
        citations.append(response.source_nodes[cit].get_text()+"\n"+meta_text)
        
    citation_text = "\n===========================================\n".join([f"{citation}" for i, citation in enumerate(citations)])
    
    # Store the citation text temporarily
    temporary_citation_text = citation_text
    
    # Return response text and make the button interactive
    return response_text, gr.update(interactive=True)

# Function to display citations when the button is clicked
def show_citations():
    global temporary_citation_text
    return temporary_citation_text

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Query")
            response_text = gr.Textbox(label="Response", interactive=False)
        with gr.Column():
            citation_button = gr.Button("Show Citations", interactive=False)
            citation_text = gr.Textbox(label="Citations", interactive=False)
    
    # Set the query input to trigger the search_query function
    query_input.submit(search_query, inputs=query_input, outputs=[response_text, citation_button])
    
    # Set the button to show the citations when clicked
    citation_button.click(show_citations, outputs=citation_text)

# Launch the interface
demo.launch(server_name="192.168.0.44")