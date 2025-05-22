import argparse
import os 

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from get_embedding import get_embedding

# --- Config
CHROMA_PATH = "chroma_langchain_db/eu-omnibus"

EMBEDDING = "snowflake-arctic-embed2"
LLM = "mistral"

PROMPT_TEMPLATE = """
Answer the question based ONLY on the following context:

{context}

---

Answer the question based ONLY on the above context: {question}
"""


# --- Functions

# query the LLM
def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The query text.")
    args = parser.parse_args()
    query = args.query
    query_rag(query)


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
    # print(prompt)

    # run the model
    model = OllamaLLM(model = LLM)
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response:{response}\n\n Sources:{sources}"
    print(formatted_response)

    return response



# execute script
if __name__ == "__main__":
    # main()
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Dev/test fallback
        query_rag("What are the most important proposed changes regarding the CSRD regulation?")



