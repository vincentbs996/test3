import streamlit as st
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK packages if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Read in the corpus
try:
    with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
        raw = fin.read().lower()
except FileNotFoundError:
    st.error("Error: 'chatbot.txt' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Tokenisation
sent_tokens = nltk.sent_tokenize(raw)  # Converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # Converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.pop()  # Remove the user input after processing
    return robo_response

# Streamlit UI Setup
st.title("ROBO: Your Friendly Chatbot")
st.write("My name is Robo. I will answer your queries about Chatbots. If you want to exit, type 'Bye'.")

# User input handling
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("You: ", key="input")

if st.button("Send") and user_input:
    if user_input.lower() == 'bye':
        st.session_state.conversation.append(("ROBO", "Bye! Take care.."))
        st.stop()
    elif user_input.lower() in ['thanks', 'thank you']:
        st.session_state.conversation.append(("ROBO", "You are welcome.."))
    else:
        if greeting(user_input) is not None:
            response_text = greeting(user_input)
        else:
            response_text = response(user_input)
        st.session_state.conversation.append(("ROBO", response_text))
    st.session_state.conversation.append(("You", user_input))

# Display conversation history
for speaker, text in st.session_state.conversation:
    st.write(f"**{speaker}:** {text}")
