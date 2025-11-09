import sys

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from get_embedding import get_embedding

# --- Config

CHROMA_PATH = "chroma_langchain_db/pav"

PROVIDER = "HF"
EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"
LLM = "qwen3:8b"

PROMPT_TEMPLATE = """
/no_think
Du bist ein Experte für die strategische Evaluation von Schweizer Agglomerationsverkehrsprogrammen. 
Dein Fokus liegt auf der Analyse und Bewertung von Programmen, die von Trägerschaften oder Agglomerationen eingereicht wurden, insbesondere in Bezug auf ihre
- Wirksamkeit
- Umsetzbarkeit 
- Effizienz 
Deine Antworten sind kontextsensibel, kritisch und evidenzbasiert sowie auf **Deutsch** formuliert. 
Nutze zur Beantwortung der Frage **nur** den folgenden Kontext:

{context}

---

**Wichtig:** Berücksichtige zusätzlich die Konversationshistorie, damit du auch auf Folgefragen des Users antworten kannst. Dies sind jeweils die drei letzten Fragen und Antworten:

{history}

---

Beantworte nun folgende Frage nur anhand des oben stehenden Kontextes und der Historie. Sobald du eine Antwort für dich gefunden hast, überprüfe diese nochmals ob diese Sinn ergibt und die Frage in ihren Facetten beantwortet: 

{question}

"""


# --- Functions

def query_rag(query: str, history: str):

    # prepare the data base
    embedding_function = get_embedding(
        provider=PROVIDER, 
        embedding_model = EMBEDDING, 
        show_progress = False
        )
    
    db = Chroma(
        collection_name = "pav", 
        persist_directory=CHROMA_PATH, 
        embedding_function = embedding_function
        )
    
    # search the data base
    results = db.similarity_search_with_score(query, k = 3)
    
    # confirm that context is not empty
    if not results:
        print("The context will be empty.")
    
    # create the context and join with query to prompt
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    prompt = prompt_template.format(
        context = context, 
        history = history, 
        question = query
        )

    # SHOW PROMPT ENABLE/DISABLE
    # print(f"Prompt sent to model:\n{prompt}")
    
    # run the model
    model = OllamaLLM(model = LLM)
    response = model.invoke(prompt)

    # extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return response, sources


# chat function for multiple questions 
def chat():
    print("Willkommen beim Agglo-Programm Chat! Zum verlassen 'exit' eintippen.")
    conversation_history = []

    while True:
        query = input("\nSie: ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Auf Wiedersehen.")
            break

        # Get the last 3 turns of conversation history
        history = "\n".join([f"User: {conversation_history[i]}" if i % 2 == 0 else f"Assistent: {conversation_history[i]}"
                    for i in range(-6, 0) if len(conversation_history) > abs(i)])


        response, sources = query_rag(query, history)

        # Update conversation history
        conversation_history.append(f"User: {query}")
        conversation_history.append(f"Assistent: {response}")

        print(f"\nLLM: {response}")
        print(f"Quellen: {sources}")


# execute
if __name__ == "__main__":
    if len(sys.argv) > 1:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("query", type=str, help="A one-time query")
        args = parser.parse_args()
        query_rag(args.query, "")
    else:
        chat() 
