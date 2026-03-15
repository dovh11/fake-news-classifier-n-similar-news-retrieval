
# 📰 Fake News Detector & Semantic Search

**Live Web App:** https://fake-news-classifier-n-similar-news-retrieval-5ffa2vlvqqytls4t.streamlit.app/

**Colab Notebook:**: https://colab.research.google.com/drive/11kmVPUFiapPsyxTU-2N2Beh8EwTSxX7L?usp=sharing

This project is an interactive web application that leverages foundational Natural Language Processing (NLP) techniques to classify news articles as **Real** or **Fake**. Additionally, it acts as a semantic search engine, retrieving the top 3 most similar articles from the database based on the user's input.

This project was built strictly using foundational NLP techniques (Static Word Embeddings, Mean-Pooling, and Multilayer Perceptrons) without relying on modern Transformers or Large Language Models (LLMs).

## ✨ Features

1.  **Fake News Classification**: Uses a PyTorch-based Feedforward Neural Network (MLP) to predict the authenticity of a news article.
    
2.  **Semantic Information Retrieval**: Uses Cosine Similarity to find and display the top 3 closest matching articles from the dataset.
    
3.  **Data Leakage Prevention**: Automatically cleans publisher watermarks (e.g., _"WASHINGTON (Reuters) - "_) from the dataset to ensure the model learns true linguistic semantics rather than memorizing publisher tags.
    
4.  **Interactive UI**: Built with Streamlit for a seamless, real-time user experience.
    

## 🧠 Methodology & Architecture

-   **Text Preprocessing:** Tokenization via `NLTK`, converting to lowercase, and removing non-alphanumeric characters.
    
-   **Feature Extraction (Word Embeddings):** Utilizes pre-trained 50-dimensional **GloVe** vectors (`glove-wiki-gigaword-50`) loaded via `gensim`.
    
-   **Document Representation:** Represents entire articles by taking the average (**mean-pooling**) of the word embeddings of all valid tokens within the text.
    
-   **Information Retrieval:** Calculates the **Cosine Similarity** (via `scikit-learn`) between the mean-pooled vector of the user's query and the vectors of all documents in the corpus.
    
-   **Classification Model:** A custom PyTorch Multilayer Perceptron (MLP) with one hidden layer (32 neurons, ReLU activation) optimized using Adam and Cross-Entropy Loss.
    

## 📂 Repository Structure

```
├── app.py                 # Main Streamlit application code
├── requirements.txt       # Python dependencies 
├── README.md              # Project documentation
└── data/                  # Dataset folder (ignored in git if too large)
    ├── True.csv (or .zip) # Real news dataset
    └── Fake.csv (or .zip) # Fake news dataset

```

## 📊 Dataset

This project uses the well-known **Fake News Detection Dataset** from Kaggle.

-   **Source:** https://www.kaggle.com/code/therealsampat/fake-news-detection/input
    
-   **Handling Large Files:** The app is configured to natively read both standard `.csv` files and compressed `.zip` files to bypass GitHub's 25MB file upload limit.
    

## 🚀 How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/dovh11/fake-news-classifier-n-similar-news-retrieval.git
cd fake-news-classifier-n-similar-news-retrieval

```

### 2. Prepare the Data

Ensure you have downloaded `True.csv` and `Fake.csv` from Kaggle and placed them inside a folder named `data/` in the root directory. _(You can zip them into `True.zip` and `Fake.zip` to save space)._

### 3. Install Dependencies

It is recommended to use a virtual environment. Install the required packages via `pip`:

```
pip install -r requirements.txt

```

_Note: The `requirements.txt` specifically fetches the CPU-only version of PyTorch to save space and pins `numpy<2.0.0` and `pandas<3.0.0` for compatibility._

### 4. Launch the App

Run the following command to start the Streamlit server:

```
streamlit run app.py

```

The app will automatically open in your default web browser at `http://localhost:8501`.

_(Note: The first time you run the app, it will take a minute or two to download the GloVe embeddings and train the PyTorch model. Subsequent queries will be instantaneous thanks to Streamlit's caching)._

## 🛠️ Tech Stack

-   **Python 3.11/3.12**
    
-   **PyTorch** (Neural Network Classification)
    
-   **Gensim** (GloVe Word Embeddings)
    
-   **NLTK** (Tokenization)
    
-   **Scikit-Learn** (Cosine Similarity)
    
-   **Pandas & NumPy** (Data Manipulation)
    
-   **Streamlit** (Web UI & Caching)
