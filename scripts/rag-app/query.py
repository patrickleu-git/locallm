import sys

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from get_embedding import get_embedding

# --- Config

CHROMA_PATH = "chroma_langchain_db/eu-omnibus"

EMBEDDING = "snowflake-arctic-embed2"
LLM = "mistral"

PROMPT_TEMPLATE = """
You are a legal expert. You are especially knowledgeable in sustainability reporting laws. 
Answer the question based ONLY on the following context:

{context}

---

Answer the question based ONLY on the above context: {question}
"""


# --- Functions

def query_rag(query: str):

    # prepare the data base
    embedding_function = get_embedding(embedding = EMBEDDING)
    db = Chroma(collection_name = "omnibus", persist_directory=CHROMA_PATH, embedding_function = embedding_function)
    
    # search the data base
    results = db.similarity_search_with_score(query, k = 5)
    
    # confirm that context is not empty
    if not results:
        print("The context will be empty.")
    
    # create the context and join with query to prompt
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context, question = query)

    # run the model
    model = OllamaLLM(model = LLM)
    response = model.invoke(prompt)

    # extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return response, sources


# chat function for multiple questions 
def chat():
    print("Welcome to RAG chat! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response, sources = query_rag(query)
        
        print(f"\nLLM: {response}")
        print(f"Sources: {sources}")

# execute
if __name__ == "__main__":
    if len(sys.argv) > 1:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("query", type=str, help="A one-time query")
        args = parser.parse_args()
        query_rag(args.query)
    else:
        chat() 
