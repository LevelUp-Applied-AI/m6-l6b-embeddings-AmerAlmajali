"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    # FIX: was iterating over characters; must split into words first
    vectors = [embeddings[w] for w in text.lower().split() if w in embeddings]
    if not vectors:
        return np.zeros(50)  # dim = 50 fixed by the provided 50d GloVe file
    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    # FIX: torch imported at top level so this function works outside __main__
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    # Expand attention mask to match hidden state dimensions
    mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
    # Zero out padding token hidden states
    masked_hidden = hidden_states * mask
    # Sum and divide by number of real (non-padding) tokens
    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1)
    embedding = summed / counts

    return embedding.squeeze().numpy()


def compare_similarities(
    texts, queries, tfidf_sim, glove_embeddings, bert_model, bert_tokenizer
):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """
    text_to_idx = {t: i for i, t in enumerate(texts)}
    results = {}

    # Precompute GloVe & BERT embeddings for the full corpus once
    glove_corpus = np.stack([text_to_glove(t, glove_embeddings) for t in texts])

    bert_corpus = []
    bert_model.eval()
    for t in texts:
        bert_corpus.append(extract_bert_embedding(t, bert_tokenizer, bert_model))
    bert_corpus = np.stack(bert_corpus)  # (n, 768)

    for query in queries:
        q_idx = text_to_idx.get(query)

        # ── TF-IDF ──────────────────────────────────────────────────────────
        tfidf_scores = tfidf_sim[q_idx].copy()
        tfidf_scores[q_idx] = -1  # exclude self
        top3_tfidf_idx = np.argsort(tfidf_scores)[::-1][:3]
        top3_tfidf = [(texts[i], float(tfidf_scores[i])) for i in top3_tfidf_idx]

        # ── GloVe ────────────────────────────────────────────────────────────
        q_glove = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(q_glove, glove_corpus)[0]
        glove_scores[q_idx] = -1
        top3_glove_idx = np.argsort(glove_scores)[::-1][:3]
        top3_glove = [(texts[i], float(glove_scores[i])) for i in top3_glove_idx]

        # ── BERT ─────────────────────────────────────────────────────────────
        q_bert = extract_bert_embedding(query, bert_tokenizer, bert_model).reshape(
            1, -1
        )
        bert_scores = sklearn_cosine(q_bert, bert_corpus)[0]
        bert_scores[q_idx] = -1
        top3_bert_idx = np.argsort(bert_scores)[::-1][:3]
        top3_bert = [(texts[i], float(bert_scores[i])) for i in top3_bert_idx]

        results[query] = {
            "tfidf": top3_tfidf,
            "glove": top3_glove,
            "bert": top3_bert,
        }

    return results


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # OOV rate (Task 5 analysis)
    all_words = " ".join(texts).lower().split()
    unique_words = set(all_words)
    oov = unique_words - set(glove.keys())
    oov_rate = len(oov) / len(unique_words) * 100
    print(
        f"OOV rate: {oov_rate:.1f}% ({len(oov):,} / {len(unique_words):,} unique tokens)"
    )
    print(f"Sample OOV words: {sorted(list(oov))[:20]}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [
            df[df["category"] == cat]["text"].iloc[0] for cat in df["category"].unique()
        ]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in comparison.keys():
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")
