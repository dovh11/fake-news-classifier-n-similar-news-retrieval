import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import gensim.downloader as api
import streamlit as st

# Suppress warnings and download NLTK tokenizers
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# 1. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detector & Semantic Search")
st.markdown("Enter a news article below to classify it as **Real** or **Fake** and find similar articles in our database.")

EMBEDDING_DIM = 50

# ==========================================
# 2. CACHED MODEL & DATA LOADING
# ==========================================
# @st.cache_resource ensures these functions only run ONCE when the app starts.
# It prevents the app from retraining the model every time the user clicks a button!

@st.cache_resource(show_spinner="Loading GloVe Embeddings...")
def load_embeddings():
    return api.load("glove-wiki-gigaword-50")

def tokenize_text(text):
    return [t for t in word_tokenize(text.lower()) if t.isalnum()]

def get_document_embedding(text, embeddings, dim):
    tokens = tokenize_text(text)
    valid_vectors = [embeddings[token] for token in tokens if token in embeddings]
    if not valid_vectors:
        return np.zeros(dim)
    return np.mean(valid_vectors, axis=0)

@st.cache_resource(show_spinner="Loading and Preprocessing Dataset...")
def load_and_embed_data(_embeddings):
    # Dynamically check for standard or zipped CSVs to bypass GitHub 25MB limits
    true_path = 'True.zip' if os.path.exists('True.zip') else ('True.csv.zip' if os.path.exists('True.csv.zip') else 'True.csv')
    fake_path = 'Fake.zip' if os.path.exists('Fake.zip') else ('Fake.csv.zip' if os.path.exists('Fake.csv.zip') else 'Fake.csv')
    
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        st.error("Dataset missing! Please ensure True.csv/Fake.csv (or their .zip versions) are in the folder.")
        st.stop()
        
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    
    # Remove data leakage watermark
    df_true['text'] = df_true['text'].apply(
        lambda x: x.split(' - ', 1)[1] if isinstance(x, str) and ' - ' in x[:100] else x
    )
    
    df_true['label'] = 1
    df_fake['label'] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.dropna(subset=['text', 'label'])
    
    # Sample down to 5000 rows for faster web app startup time. 
    # You can increase this, but the initial load will take longer!
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    # Generate embeddings
    df['embedding'] = df['text'].apply(lambda x: get_document_embedding(x, _embeddings, EMBEDDING_DIM))
    return df

class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

@st.cache_resource(show_spinner="Training PyTorch Model...")
def train_model(_df):
    X = np.vstack(_df['embedding'].values)
    y = _df['label'].values
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    model = FakeNewsClassifier(input_dim=EMBEDDING_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model

# Initialize everything
glove_vectors = load_embeddings()
df = load_and_embed_data(glove_vectors)
model = train_model(df)

# ==========================================
# 3. INTERACTIVE UI LOGIC
# ==========================================
st.write("---")
user_input = st.text_area("Paste news text here:", height=200)

if st.button("Analyze Article", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # 1. Embed user query
            query_emb = get_document_embedding(user_input, glove_vectors, EMBEDDING_DIM)
            query_tensor = torch.FloatTensor(query_emb).unsqueeze(0)
            
            # 2. Predict with PyTorch Model
            with torch.no_grad():
                outputs = model(query_tensor)
                _, prediction = torch.max(outputs, 1)
                is_real = prediction.item() == 1
            
            # 3. Display Classification Result
            st.subheader("Classification Result")
            if is_real:
                st.success("✅ This article appears to be **REAL NEWS**.")
            else:
                st.error("🚨 This article appears to be **FAKE NEWS**.")
                
            # 4. Retrieve Similar Articles
            st.subheader("🔍 Top 5 Similar Articles in Database")
            doc_embs = np.vstack(df['embedding'].values)
            similarities = cosine_similarity(query_emb.reshape(1, -1), doc_embs)[0]
            top_indices = np.argsort(similarities)[::-1][:5]
            
            for i, idx in enumerate(top_indices):
                sim_score = similarities[idx]
                sim_text = df.iloc[idx]['text']
                sim_label = "Real" if df.iloc[idx]['label'] == 1 else "Fake"
                
                # Use Streamlit expanders to make the UI look clean
                with st.expander(f"Match #{i+1} | Label: {sim_label} | Similarity: {sim_score:.2f}"):
                    st.write(sim_text)
