# Steam Reviews NLP

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-DistilBERT-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers)
[![BERTopic](https://img.shields.io/badge/BERTopic-topic%20modelling-4B8BBE)](https://maartengr.github.io/BERTopic/)
[![Gemini](https://img.shields.io/badge/Gemini-aspect%20extraction-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20baseline-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end NLP study of 434,891 Steam game reviews, reading the same corpus four ways (classification, fine-tuning, topic modelling, aspect-based sentiment) and weighing, at each step, what each method adds against what it costs to run.

---

## Objective

An end-to-end NLP analysis of Steam user reviews covering top-selling video games. The goal is to measure what players actually say, how reliably a model can tell praise from complaint, and which recurring themes drive satisfaction or frustration across major titles.

A game review is rarely one-dimensional: a player can praise the gameplay and condemn the netcode in the same paragraph. A single *Recommended / Not Recommended* vote flattens that, and so does a single-label classifier or a single-topic model. The project answers the three questions below by reading the same corpus four ways, comparing each method not only on accuracy but on the volume at which it is worth running.

---

## Questions this project answers

| Question | Answered in | Short answer |
| --- | --- | --- |
| **What do players actually say?** | 01, 05, 06 | Reviews are genuinely multi-aspect (2.36 verdicts per review); themes split into stable families — 14 to 24 complaint topics and 36 praise topics across runs (BERTopic's exact count varies; see below). |
| **How reliably can a model tell praise from complaint?** | 03, 04 | A fine-tuned DistilBERT reaches **0.886 accuracy / 0.873 macro F1**, well above a zero-shot transformer (0.750) and a majority-class floor (0.678). |
| **Which themes drive satisfaction or frustration across titles?** | 05, 06 | Gameplay drives satisfaction (50% positive); `price_value` (82% negative) and `performance` (81% negative) drive frustration, consistently across the top games. |

---

## Dataset

- **Source**: [Steam Reviews Dataset](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) (Luthfi Mahendra, Kaggle), public.
- **Size**: 434,891 reviews across 8 columns, dominated by four titles (PUBG, GTA V, Rust, Rocket League).
- **Label**: each review carries the player's own *Recommended / Not Recommended* vote, which supervises the classifiers without any manual labelling. Base rate after cleaning: 67.8% Recommended.
- **Access**: download from Kaggle and place `steam_reviews.csv` on Google Drive (see Reproduce). Raw and processed data are not versioned (see `.gitignore`).
- **Known data-quality gap**: 12.6% of texts are repeated and 3.65% carry a free-product promotional notice, both handled in notebook 02.

---

## Project Structure

```
steam-reviews-nlp/
│
├── 01_data_ingestion_eda.ipynb             # ingestion, quality audit, EDA
├── 02_preprocessing_pipeline.ipynb         # cleaning, boilerplate removal, feature engineering
├── 03_baseline_sentiment_pretrained.ipynb  # zero-shot DistilBERT baseline + error analysis
├── 04_finetuning_benchmark.ipynb           # fine-tuning + three-way benchmark + cost
├── 05_topic_modelling_bertopic.ipynb       # unsupervised topic discovery (BERTopic)
├── 06_aspect_based_sentiment.ipynb         # LLM aspect-based sentiment (Gemini)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproduce

The notebooks were built for **Google Colab** with data on **Google Drive**. The fine-tuning step needs a GPU, so Colab is the target rather than a local machine.

Each notebook resolves its data directory from the `STEAM_ROOT` environment variable, falling back to the Drive path they were written against. Set it before running the first cell if your data lives elsewhere:

```python
import os
os.environ["STEAM_ROOT"] = "/content/drive/MyDrive/<your path>"
```

The Drive mount is skipped automatically outside Colab, so the notebooks also open and run against a local directory. Notebook 04 still carries the Drive path in its first cell and will get the same treatment on its next run.

### 1. Clone

```bash
git clone https://github.com/juliettebm/steam-reviews-nlp.git
cd steam-reviews-nlp
```

### 2. Data

Download the [dataset](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) and place it on Drive:

```
MyDrive/Colab Notebooks/Projets/steam_reviews/data/raw/steam_reviews.csv
```

### 3. Run

Open each notebook in Colab and run in order (01 to 06). Notebook 02 writes the processed corpus that 03 to 06 consume. Dependencies install from the notebooks themselves (`!pip install ...`); `requirements.txt` lists them for a local run.

Notebook 06 calls the Gemini API: add a `GEMINI_API_KEY` in the **Colab Secrets** panel () with notebook access enabled. Its free tier is rate-limited (~5 requests/min), so the extraction is paced accordingly.

---

## Methodology

1. **Ingestion and audit** (01): load 434,891 reviews, quantify duplicates, missing values, repeated texts and promotional notices, then univariate and bivariate EDA (Mann-Whitney tests on playtime, length, helpfulness against the recommendation).
2. **Preprocessing** (02): drop empty reviews, strip the free-product notice, deduplicate long repeats, filter very short reviews, build two cleaning strategies (`review_raw` minimal, `review_clean` heavy), and engineer features (playtime tiers, engagement flags).
3. **Baseline** (03): zero-shot DistilBERT (SST-2) on a stratified 5,000-review evaluation set, with a diagnostic error analysis on length, technical vocabulary and sarcasm.
4. **Fine-tuning and benchmark** (04): a TF-IDF + logistic-regression baseline and a fine-tuned DistilBERT, compared against the zero-shot model and a majority-class floor on the same evaluation set, plus a cost comparison. The training sample is split three ways: 30,000 reviews to train, 3,000 held out to select the checkpoint, and the 5,000-review set from step 3 kept for reporting only. Exclusion from the training pool is done on the review text rather than on the index, because 12.64% of reviews share their text with another row and an index-based split would not catch them.
5. **Topic modelling** (05): BERTopic (sentence embeddings + UMAP + HDBSCAN) run separately on complaint and praise sub-corpora, read against game and playtime metadata.
6. **Aspect-based sentiment** (06): Gemini with structured output extracts, per review, each of six aspects and its sentiment, with an evidence-quote verification step before aggregation.

---

## Key Results

### Classification benchmark (5,000-review reporting set)

This set selects nothing. Checkpoint selection runs on a separate 3,000-review validation slice taken from the training sample, so the figures below are measured on data that played no part in choosing the model. These are the numbers from that protocol, not the earlier run where the reporting set doubled as the checkpoint-selection set.

| Model | Accuracy | Macro F1 |
| --- | :---: | :---: |
| Majority class | 0.678 | 0.404 |
| TF-IDF + Logistic Regression | 0.843 | 0.826 |
| DistilBERT (zero-shot, SST-2) | 0.750 | 0.742 |
| **DistilBERT (fine-tuned)** | **0.886** | **0.873** |

The zero-shot transformer is *beaten by a plain TF-IDF + LogReg*: pretrained sentiment does not transfer cleanly to Steam's irony and slang. Fine-tuning (6.3 min on a T4) closes the gap and wins overall, fixing 858 evaluation reviews while breaking 176, a net gain of 682. The fix moved these numbers by about 0.2 points of accuracy from the earlier, leakage-affected run — small in this case, because two epochs left little room for the checkpoint choice to overfit, but the protocol is now honest regardless of how large the effect turned out to be.

### Topic modelling

Run on ~20,000 complaints and ~20,000 praise reviews: **complaint topics vary from run to run (14–24 observed, 24 in the committed run, 39.1% unassigned)** — UMAP and HDBSCAN are not fully seed-stable, and notebook 05 documents this rather than papering over it — against a stable **36 praise topics** (42.6% unassigned). 40,012 topic-annotated reviews exported.

### Aspect-based sentiment (targeted 400-review sample)

| Metric | Value |
| --- | --- |
| Aspect mentions extracted | **943** |
| Mentions per review | **2.36** |
| Failed API calls | 0 |
| Evidence quotes found verbatim | **98.5%** |
| Reviews mixing positive and negative aspects | 17.6% |

Gameplay is the most-discussed aspect and skews positive (50%); `price_value` (82% negative) and `performance` (81% negative) are the sharpest pain points. The recurring pattern is gameplay praised against multiplayer or performance criticised, a "want to like it but can't fully" profile the single-label methods could not represent.

---

## Limitations

- The aspect-based sample (notebook 06) is **targeted, not representative**: 400 longer reviews, 60% complaints. Comparisons between aspects hold; absolute rates do not describe the corpus.
- The LLM extraction is checked for fabricated quotes but **not validated against human annotation**, so its output is plausible and internally consistent rather than measured against ground truth.
- Per-game and per-tier breakdowns rest on a few dozen mentions each; read the pattern, not the digits.

---

## Disclaimer

Educational project on a public dataset of user-generated game reviews. Used for analysis and demonstration only.

---

## Stack

Python 3.11 · pandas · scikit-learn · Hugging Face Transformers · PyTorch · BERTopic · Sentence-Transformers · Google Gemini · Matplotlib / Seaborn

---

## License

Released under the [MIT License](LICENSE).

---

## Author

**Juliette Bouli-Mengue**
Clinical Research to Data Science
