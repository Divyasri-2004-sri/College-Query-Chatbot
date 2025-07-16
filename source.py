import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch

# Use smaller & faster model
gen_model = "google/flan-t5-small"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
tokenizer = AutoTokenizer.from_pretrained(gen_model)
model = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(device)
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# College knowledge base
knowledge_base = [
    "Our college offers B.Tech, MBA, and BBA courses.",
    "Hostel fees are 50,000 INR per year.",
    "Admissions start every June.",
    "Scholarships are available for the top 10% of students.",
    "The campus has modern labs and sports facilities.",
]

# Precompute embeddings
kb_embeddings = embed_model.encode(knowledge_base, convert_to_tensor=True)

# Small talk
simple_responses = {
    "hi": "Hello! How can I help you with your college queries today?",
    "hello": "Hi there! What do you want to know about the college?",
    "hey": "Hey! Ask me anything about the college.",
    "thanks": "You're welcome!",
    "thank you": "Glad to help!",
    "bye": "Goodbye! Have a great day.",
    "goodbye": "See you later!",
    "how are you": "I'm fine, glad to see you!",
    "how r u": "I'm good, what about you?",
}

def get_relevant_info(question, top_k=2):
    q_embed = embed_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_embed, kb_embeddings, top_k=top_k)[0]
    return "\n".join(knowledge_base[hit['corpus_id']] for hit in hits)

def ask_bot(question):
    q_lower = question.strip().lower()
    if q_lower in simple_responses:
        return simple_responses[q_lower]

    info = get_relevant_info(question)
    prompt = f"""
You are a helpful college assistant. Use ONLY the information below to answer the question.
If the information does not contain the answer, say "Sorry, I don't know."

Information:
{info}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output = model.generate(**inputs, max_length=100, no_repeat_ngram_size=3)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    if "Sorry" in answer or not answer:
        return "I'm not sure about that. Please contact the college office for more details."
    return answer

# Gradio Interface
def chatbot_ui(user_input):
    return ask_bot(user_input)

interface = gr.Interface(
    fn=chatbot_ui,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the college..."),
    outputs="text",
    title="🎓 College Chatbot",
    description="Ask anything about our college!",
    theme="default",
    allow_flagging="never",
)

interface.launch(share=True)