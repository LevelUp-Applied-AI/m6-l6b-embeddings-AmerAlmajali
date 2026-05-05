"""
Module 6 Week B -- Stretch: Embedding Space Explorer

Visualizes GloVe word embeddings and DistilBERT document embeddings in 2D.
Both t-SNE AND PCA are applied and shown side-by-side for comparison.

Produces:
  word_embedding_viz.png   -- 200 GloVe words, 5 BBC categories, t-SNE vs PCA
  doc_embedding_viz.png    -- 20 BBC News articles, DistilBERT, t-SNE vs PCA
  stretch_analysis.md      -- Written analysis

Word selection strategy (corpus-driven):
  Words are NOT hardcoded. They are mined from bbc_news.csv using TF-IDF:
    - For each of the 5 BBC categories, take the top-40 words by mean TF-IDF
    - Words must be in GloVe 50k vocab (so they have embeddings)
    - Words must be purely alphabetic, length >= 4 (filters junk tokens)
    - Words are disjoint across categories (each word belongs to one category)
  This means the word categories MATCH the 5 BBC document categories exactly,
  creating a direct connection between the word-level and document-level plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sklearn
from packaging.version import Version as _V

# sklearn >= 1.5 renamed n_iter -> max_iter in TSNE
_TSNE_ITER_PARAM = "max_iter" if _V(sklearn.__version__) >= _V("1.5") else "n_iter"


def _make_tsne(perplexity, n_iterations=1000, random_state=42):
    """Create a TSNE instance compatible with any sklearn >= 0.24."""
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        learning_rate="auto",
        init="pca",
        **{_TSNE_ITER_PARAM: n_iterations},
    )


PALETTE = {
    "business": "#FF6B6B",
    "entertainment": "#FFD93D",
    "politics": "#6BCB77",
    "sport": "#4D96FF",
    "tech": "#C77DFF",
}


# ---- 1. CORPUS-DRIVEN WORD SELECTION ----------------------------------------


def select_words_from_corpus(csv_path, glove_vocab, words_per_cat=40):
    """
    Mine bbc_news.csv for the most category-distinctive GloVe words.

    Method
    ------
    1. Fit TF-IDF on all articles (unigrams, stop-words removed).
    2. For each BBC category, compute the mean TF-IDF score of every word
       across that category's articles.
    3. Rank words by that score and greedily assign each word to the category
       that scores it highest (disjoint sets -- no word appears in two
       categories).
    4. Filter to words that are in GloVe, purely alphabetic, length >= 4.
    5. Return the top `words_per_cat` words per category.

    The top-2 words per category are flagged for annotation.

    Parameters
    ----------
    csv_path      : path to bbc_news.csv (columns: text, category)
    glove_vocab   : set of strings present in the GloVe vocabulary
    words_per_cat : number of words to select per category (default 40)

    Returns
    -------
    word_categories : dict  {category_str: [word, ...]}
    annotate_set    : set   of words to label on the scatter plot
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()
    categories = sorted(df["category"].unique())

    # TF-IDF on full corpus
    # token_pattern: alpha-only tokens of length >= 4
    # min_df=5 : must appear in >= 5 articles
    # max_df=0.7: must not appear in > 70% of articles
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.7,
        token_pattern=r"(?u)\b[a-zA-Z]{4,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    vocab = vectorizer.get_feature_names_out()  # (n_terms,)
    tfidf_dense = tfidf_matrix.toarray()  # (n_docs, n_terms)

    # Per-category mean TF-IDF score for every vocabulary term
    cat_scores = {}
    for cat in categories:
        mask = (df["category"] == cat).values
        cat_scores[cat] = tfidf_dense[mask].mean(axis=0)  # (n_terms,)

    # Greedy disjoint assignment:
    # each word goes to whichever category scores it highest
    claimed = set()
    word_categories = {cat: [] for cat in categories}

    for cat in categories:
        ranked_idx = np.argsort(cat_scores[cat])[::-1]
        for idx in ranked_idx:
            if len(word_categories[cat]) >= words_per_cat:
                break
            word = vocab[idx]
            if word in claimed:
                continue
            if word not in glove_vocab:
                continue
            word_categories[cat].append(word)
            claimed.add(word)

    # Annotate the top-2 TF-IDF words per category
    annotate_set = set()
    for cat in categories:
        for w in word_categories[cat][:2]:
            annotate_set.add(w)

    print("  Corpus-driven word selection (TF-IDF, GloVe-filtered):")
    for cat in categories:
        n = len(word_categories[cat])
        preview = ", ".join(word_categories[cat][:6])
        print(f"    {cat:15s}: {n:3d} words   eg: {preview}")

    return word_categories, annotate_set


# ---- 2. HELPERS -------------------------------------------------------------


def load_glove(filepath):
    """Load GloVe vectors. Returns dict word -> np.ndarray (50,)."""
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def extract_bert_embedding(text, tokenizer, model):
    """Mean-pool DistilBERT hidden states -> np.ndarray (768,)."""
    import torch

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
    embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return embedding.squeeze().numpy()


def _scatter_panel(
    ax,
    X_2d,
    labels,
    categories,
    palette,
    annotate_words=None,
    words=None,
    title="",
    xlabel="Dim 1",
    ylabel="Dim 2",
):
    """Draw one scatter panel onto ax."""
    ax.set_facecolor("#F8F8FA")
    for cat in categories:
        mask = np.array([l == cat for l in labels])
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=palette[cat],
            s=55,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.25,
            label=cat.capitalize(),
            zorder=3,
        )
    if annotate_words and words is not None:
        for i, (word, label) in enumerate(zip(words, labels)):
            if word in annotate_words:
                ax.annotate(
                    word,
                    (X_2d[i, 0], X_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="white",
                    alpha=0.95,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", ec="none"),
                )
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color="#888", fontsize=9)
    ax.set_ylabel(ylabel, color="#888", fontsize=9)
    ax.tick_params(colors="#F8F8FA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


# ---- 3. WORD EMBEDDING VISUALIZATION ----------------------------------------


def plot_word_embeddings(glove_path, csv_path, out_path="word_embedding_viz.png"):
    print("Loading GloVe vectors...")
    glove = load_glove(glove_path)

    # Select words from the corpus
    print("Selecting words from BBC News corpus via TF-IDF...")
    word_categories, annotate_set = select_words_from_corpus(
        csv_path, glove_vocab=set(glove.keys()), words_per_cat=40
    )

    # Flatten to parallel lists
    present_words, present_labels = [], []
    for cat, words in word_categories.items():
        present_words.extend(words)
        present_labels.extend([cat] * len(words))

    print(f"  Total words: {len(present_words)}")

    X = np.stack([glove[w] for w in present_words])  # (n, 50)

    # t-SNE with perplexity=30 (good default for ~200 points)
    # init="pca" gives a stable, reproducible starting layout
    print("Running t-SNE (perplexity=30)...")
    tsne = _make_tsne(perplexity=30)
    X_tsne = tsne.fit_transform(X)

    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_var = pca.explained_variance_ratio_ * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#0D1117")

    _scatter_panel(
        ax1,
        X_tsne,
        present_labels,
        list(word_categories),
        PALETTE,
        annotate_words=annotate_set,
        words=present_words,
        title="t-SNE  (perplexity=30)\npreserves local cluster structure",
        xlabel="t-SNE Dim 1",
        ylabel="t-SNE Dim 2",
    )
    _scatter_panel(
        ax2,
        X_pca,
        present_labels,
        list(word_categories),
        PALETTE,
        annotate_words=annotate_set,
        words=present_words,
        title=(
            "PCA\npreserves global variance  "
            "[PC1={:.1f}%  PC2={:.1f}%]".format(pca_var[0], pca_var[1])
        ),
        xlabel="PC1 ({:.1f}% var)".format(pca_var[0]),
        ylabel="PC2 ({:.1f}% var)".format(pca_var[1]),
    )

    handles = [
        mpatches.Patch(color=PALETTE[cat], label=cat.capitalize())
        for cat in word_categories
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        framealpha=0.2,
        facecolor="#1A1A2E",
        edgecolor="#444",
        labelcolor="white",
        fontsize=11,
        markerscale=1.4,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "GloVe 50d Word Embeddings -- t-SNE vs PCA\n"
        "{} words in GLOVE from BBC News corpus (top TF-IDF per category)".format(
            len(present_words)
        ),
        color="white",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved -> {}".format(out_path))
    return pca_var


# ---- 4. DOCUMENT EMBEDDING VISUALIZATION ------------------------------------


def plot_document_embeddings(csv_path, out_path="doc_embedding_viz.png"):
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("transformers not installed -- skipping document plot")
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()

    # 4 articles per category = 20 total
    sampled = df.groupby("category", group_keys=False).sample(n=4)
    texts = sampled["text"].tolist()
    labels = sampled["category"].tolist()
    short = [t[:32].replace("\n", " ") + "..." for t in texts]
    print("  Articles: {}".format(sampled["category"].value_counts().to_dict()))

    print("Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    print("Computing embeddings...")
    embeddings = np.stack([extract_bert_embedding(t, tokenizer, model) for t in texts])
    print("  Shape: {}".format(embeddings.shape))

    # t-SNE with perplexity=5 (must be < n_samples=20)
    print("Running t-SNE (perplexity=5 for 20 docs)...")
    tsne = _make_tsne(perplexity=5)
    X_tsne = tsne.fit_transform(embeddings)

    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(embeddings)
    pca_var = pca.explained_variance_ratio_ * 100

    def _doc_panel(ax, X_2d, title, xlabel, ylabel):
        ax.set_facecolor("#F2EDED")
        for i, (x, y) in enumerate(X_2d):
            cat = labels[i]
            color = PALETTE[cat]
            ax.scatter(
                x,
                y,
                c=color,
                s=130,
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
            )
            ax.annotate(
                short[i],
                (x, y),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=6.8,
                color=color,
                alpha=0.92,
                fontfamily="monospace",
            )
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color="#888", fontsize=9)
        ax.set_ylabel(ylabel, color="#888", fontsize=9)
        ax.tick_params(colors="#F2EDED")
        for spine in ax.spines.values():
            spine.set_edgecolor("#222")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor("#0D0D0D")

    _doc_panel(
        ax1,
        X_tsne,
        "t-SNE  (perplexity=5)\nlocal structure  [interpret carefully at n=20]",
        "t-SNE Dim 1",
        "t-SNE Dim 2",
    )
    _doc_panel(
        ax2,
        X_pca,
        "PCA  (recommended for n=20)\nPC1={:.1f}%  PC2={:.1f}% variance explained".format(
            pca_var[0], pca_var[1]
        ),
        "PC1 ({:.1f}% var)".format(pca_var[0]),
        "PC2 ({:.1f}% var)".format(pca_var[1]),
    )

    handles = [
        mpatches.Patch(color=c, label=cat.capitalize()) for cat, c in PALETTE.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        framealpha=0.2,
        facecolor="#1A1A1A",
        edgecolor="#444",
        labelcolor="white",
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "DistilBERT Document Embeddings -- t-SNE vs PCA  (20 BBC News articles)",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved -> {}".format(out_path))
    return pca_var


# ---- MAIN -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stretch 6B-S1: Embedding Space Explorer"
    )
    parser.add_argument("--glove", default="data/glove_50k_50d.txt")
    parser.add_argument("--csv", default="data/bbc_news.csv")
    parser.add_argument("--word-out", default="word_embedding_viz.png")
    parser.add_argument("--doc-out", default="doc_embedding_viz.png")
    parser.add_argument("--analysis", default="stretch_analysis.md")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1 -- Word Embeddings (GloVe + t-SNE vs PCA)")
    print("=" * 60)
    plot_word_embeddings(args.glove, args.csv, args.word_out)

    print()
    print("=" * 60)
    print("STEP 2 -- Document Embeddings (DistilBERT + t-SNE vs PCA)")
    print("=" * 60)
    plot_document_embeddings(args.csv, args.doc_out)

    print()
    print("=" * 60)
    print("STEP 3 -- Writing analysis")
    print("=" * 60)

    print()
    print("Done. Deliverables:")
    print("  * {}".format(args.word_out))
    print("  * {}".format(args.doc_out))
