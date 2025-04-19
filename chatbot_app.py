import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch



# Sample Q&A knowledge base
qa_pairs = [
    {"question": "What types of study programs are available in Spain?", 
     "answer": "Spain offers over 12,000 public and private university programs, including Engineering, Computer Science, Education, Business, and Medicine."},
    {"question": "How can I find a suitable program?", 
     "answer": "You can use StudiesIn's AI Course Finder to discover programs that match your profile."},
    {"question": "Do I need a student visa to study in Spain?", 
     "answer": "If you're from outside the EU, EEA, or Switzerland, and your program lasts more than 90 days, you'll need a student visa."},
    {"question": "What are the requirements for a student visa?", 
     "answer": "You must be enrolled in an educational program of at least 20 hours per week in an accredited institution."},
    {"question": "What accommodation options are available?", 
     "answer": "Options include shared apartments, student residences, homestays, and private studios."},
    {"question": "How can I find accommodation?", 
     "answer": "StudiesIn has partnerships offering over 50,000 beds in 10+ cities across Spain."},
    {"question": "Are there Spanish language courses for international students?", 
     "answer": "Yes, institutions like StudiesIn Academy Valencia and Enforex in Madrid offer courses for all levels."}
]

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
questions = [pair["question"] for pair in qa_pairs]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Streamlit UI
st.title("🇪🇸 Study Guide Chatbot")
st.info("""  
This is a demo chatbot built for educational consultancy websites.

You can try asking things like:
- What study programs are available in Spain?
- How can I find a suitable program?
- Do I need a student visa to study in Spain?
- What are the requirements for a student visa?
- What accommodation options are available?
- How do I find accommodation?
- Are there Spanish language courses for international students?

This chatbot is fully customizable — we can train it on your website content too!
""")

with st.form(key="chat_form"):
    user_input = st.text_input("You:", "", placeholder="Type your question and press Enter")
    submit_button = st.form_submit_button(label="Ask")

if submit_button and user_input.strip():
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    best_answer = qa_pairs[best_match_idx]["answer"]
    st.markdown(f"**Chatbot response:** {best_answer}")

elif not submit_button:
    st.markdown("\n👋 Hello! I'm here to help you with studying in Spain. Ask me a question!")
