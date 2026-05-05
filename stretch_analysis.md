# Stretch 6B-S1 — Embedding Space Analysis

## Word Selection Method

Words were selected programmatically from `bbc_news.csv` using TF-IDF scores
rather than being hardcoded. For each of the 5 BBC News categories (business,
entertainment, politics, sport, tech), the script computes the mean TF-IDF
score of every vocabulary token across that category's articles, then greedily
assigns each token to the category that scores it highest (producing disjoint
sets — no word appears in two categories). Tokens must be purely alphabetic, at
least 4 characters long, and present in the GloVe 50k vocabulary. The top 40
tokens per category are kept, giving 200 words total. This approach ensures the
word categories are the **same 5 topics** as the document-embedding task, so
both plots visualize the same semantic space — one at word granularity, one at
document granularity.

Top words selected per category:

| Category      | Top words (by TF-IDF)                                      |
|---------------|------------------------------------------------------------|
| business      | growth, company, economy, firm, year, bank                 |
| entertainment | film, best, band, festival, music, awards                  |
| politics      | labour, election, blair, party, brown, minister            |
| sport         | game, club, england, team, players, play                   |
| tech          | software, technology, mobile, users, microsoft, computer   |

---

## Dimensionality Reduction: t-SNE vs PCA

Both techniques are applied and plotted side-by-side so their trade-offs are
directly visible.

| Property              | t-SNE                          | PCA                            |
|-----------------------|--------------------------------|--------------------------------|
| Structure preserved   | Local (tight clusters)         | Global (total variance)        |
| Deterministic?        | No (random seed fixed to 42)   | Yes                            |
| Interpretable axes?   | No                             | Yes — PC1/PC2 = variance axes  |
| Reliable at n=20?     | No — perplexity must be << n   | Yes                            |

**Word embeddings (n≈200):** t-SNE with `perplexity=30` is the primary view
because it preserves local neighbourhood structure, pulling semantically similar
words together regardless of their global position. `init="pca"` is used for a
stable, reproducible starting layout. PCA is shown alongside to reveal the
global variance axes — which semantic contrasts account for the most spread in
the 50-dimensional GloVe space.

**Document embeddings (n=20):** t-SNE requires `perplexity < n`, so
`perplexity=5` is used. At this setting the algorithm essentially reflects
pairwise distances rather than manifold structure, making results less
trustworthy. The subtitle explicitly flags this limitation. PCA is the
recommended view at this sample size.

---

## Word Embedding Space (GloVe + t-SNE vs PCA)

The t-SNE plot reveals meaningful structure in the 50-dimensional GloVe space,
though with notable cross-category mixing that itself is informative. **Tech**
words (*software*, *technology*) form the most isolated cluster, pulled to the
far left of the t-SNE plot — evidence that computing vocabulary occupies a
distinct distributional region in GloVe's training corpus. **Sport** words
(*game*, *club*) form a second recognisable cluster toward the right-centre,
consistent with the narrow, event-specific register of sports reporting.

The most interesting pattern is the overlap between **business**, **politics**,
and **entertainment** words in the centre of the t-SNE plot. This is not a
failure of the embedding — it reflects a genuine distributional fact: the BBC's
business and politics coverage shares institutional vocabulary (*firm*, *party*,
*economy*, *minister*), while entertainment words like *festival* and *awards*
appear in contexts that also use evaluative language common across topics.
The annotated words *election* and *labour* sit deep inside the politics cluster,
while *growth* and *company* sit slightly apart from it, illustrating that
economic terms sit closer to the business–politics boundary than to pure finance
vocabulary.

In the PCA view, **tech** words pull to the far right of PC1 and **politics**
words pull to the bottom of PC2, suggesting PC1 captures a
technology-vs-general-language axis and PC2 captures an institutional-vs-cultural
axis. The lower total variance explained (PC1=14.4%, PC2=11.1%) confirms that
GloVe's 50-dimensional space distributes meaning across many directions — no
single axis dominates.

## Document Embedding Space (DistilBERT + t-SNE vs PCA)

The PCA plot (the more reliable view for n=20, explaining PC1=16.4% and
PC2=13.2% of variance) shows clearer category separation than the word-level
PCA, which suggests that DistilBERT's contextual, full-sentence embeddings
capture topic more crisply than averaged word vectors. **Sport** articles (*jot
joy at professional cup wi...*, *sprinter walker quits athletics...*) cluster
toward the top-left of PC1, while **business** articles (*nasdaq planning $100m
share sale...*, *saudi investor picks up the save...*) pull toward the
bottom-right — consistent with the word-level finding that sports and financial
language occupy opposite distributional poles. **Politics** articles (*clarke
defends terror detentions...*, *kilroy unveils immigration polic...*) cluster
near the centre-left, overlapping partially with sport, which reflects the
shared institutional register between political and administrative language.

**Tech** articles (*when invention turns to innovati...*, *more power to the
people says he...*) scatter across the upper-centre, some overlapping with
entertainment — unsurprising given that BBC technology articles often adopt a
consumer-facing, narrative tone similar to entertainment writing. The t-SNE
plot at perplexity=5 is visually messier as expected at n=20, but broadly
corroborates the PCA structure. Crucially, the word-level and document-level
plots tell a **consistent story**: tech is isolated, sport and business sit at
opposite ends of a domain-specificity axis, and entertainment is the most
diffuse — a pattern visible in both GloVe word space and DistilBERT document
space despite the two models operating at different granularities and being
trained with different objectives.
