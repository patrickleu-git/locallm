import gradio as gr
from query import query_rag   # adjust if your code is in another file

# Chat function
def chat_fn(message, history):
    # history comes as list of tuples [(user, assistant), ...]
    history_text = "\n".join([f"User: {m[0]}\nAssistent: {m[1]}" for m in history if m[1]])
    
    response, sources = query_rag(message, history_text)
    
    # Append Quellen nicely
    response_with_sources = f"{response}\n\n📚 Quellen: {', '.join([str(s) for s in sources if s])}"
    
    return response_with_sources

# Launch chat interface
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Agglo-Programm Chat",
    description="Stelle deine Fragen zu Agglomerationsverkehrsprogrammen!",
    examples=[
        "Wie wird die Wirksamkeit der Programme bewertet?",
        "Welche Herausforderungen bestehen bei der Umsetzung?",
        "Gibt es Unterschiede in der Effizienz zwischen Agglomerationen?"
    ],
)

if __name__ == "__main__":
    demo.launch()
