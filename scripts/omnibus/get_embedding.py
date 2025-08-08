from langchain_ollama import OllamaEmbeddings

def get_embedding(embedding: str = "snowflake-arctic-embed2"):
    """
    Function to get an embedding function.
    Useful because embedding is needed two times: (1) creating the data base and (2) embed query.
    """
    embedding = OllamaEmbeddings(
        model = embedding, 
        num_ctx = 4096, # larger context window
        temperature = 0 # no creativity
    )
    return embedding

