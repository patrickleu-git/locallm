from typing import Literal
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding(provider: Literal["Ollama", "HF"], embedding_model: str, show_progress: bool = True):
    """
    Function to get an embedding function.
    Useful because embedding is needed two times: (1) creating the data base and (2) embed query.
    """
    
    if provider == "Ollama":
        embedding = OllamaEmbeddings(
            model = embedding_model, 
            num_ctx = 4096, # larger context window
            temperature = 0 # no creativity
        )

    if provider == "HF":
        embedding = HuggingFaceEmbeddings(
            model = embedding_model,
            model_kwargs = {"device": "cpu", "trust_remote_code": True},
            show_progress = show_progress
        )

    else:
        raise ValueError("Provider must be either 'Ollama' or 'HF'")

    return embedding

