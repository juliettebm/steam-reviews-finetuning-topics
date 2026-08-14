# Steam Reviews NLP Pipeline

An end-to-end NLP study of **434,891 Steam game reviews**, reading the same corpus four different ways (classification, fine-tuning, topic modelling and aspect-based sentiment), and at each step weighing what each method adds against what it costs to run.

The [dataset](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) (Luthfi Mahendra, Kaggle) pairs each review with the player's own *Recommended / Not Recommended* vote, which supervises the classifiers without any manual labelling.

## Pipeline

The six notebooks run in order; each writes processed artifacts that the next one reads.

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | [`data_ingestion_eda`](01_data_ingestion_eda.ipynb) | Ingestion, quality audit (duplicates, missing values, system-generated reviews) and exploratory analysis. |
| 02 | [`preprocessing_pipeline`](02_preprocessing_pipeline.ipynb) | Cleaning, BBCode stripping, tokenisation, feature engineering, playtime tiers. |
| 03 | [`baseline_sentiment_pretrained`](03_baseline_sentiment_pretrained.ipynb) | Zero-shot baseline with a pretrained DistilBERT (SST-2); error analysis vs. the *Recommended* label. |
| 04 | [`finetuning_benchmark`](04_finetuning_benchmark.ipynb) | Fine-tunes DistilBERT on the corpus and benchmarks it against the baseline. |
| 05 | [`topic_modelling_bertopic`](05_topic_modelling_bertopic.ipynb) | Unsupervised topic discovery with BERTopic (sentence embeddings + UMAP + HDBSCAN). |
| 06 | [`aspect_based_sentiment`](06_aspect_based_sentiment.ipynb) | Aspect-based sentiment extraction with an LLM (Gemini, structured output). |

## Key findings

- **Fine-tuning pays off, and a simple baseline beats zero-shot.** On a 5,000-review held-out set:

  | Model | Accuracy | Macro F1 |
  |-------|:--------:|:--------:|
  | Majority class | 0.678 | 0.404 |
  | TF-IDF + Logistic Regression | 0.842 | 0.824 |
  | DistilBERT (zero-shot, SST-2) | 0.750 | 0.742 |
  | **DistilBERT (fine-tuned)** | **0.884** | **0.871** |

  The zero-shot transformer is *beaten by a plain TF-IDF + LogReg*: pretrained sentiment doesn't transfer cleanly to Steam's irony and domain slang. Fine-tuning on the corpus closes that gap and wins overall.
- **Reviews are genuinely multi-aspect.** LLM extraction on a targeted 400-review sample returned **2.36 aspect verdicts per review** (943 mentions), something the single-verdict classifier and single-topic model both had to flatten. 98.5% of the evidence quotes were found verbatim in the source, so the output rests on real text rather than paraphrase.
- **Gameplay is the draw; price and performance are the grievance.** Gameplay is the most-discussed aspect and skews positive (50%); `price_value` (82% negative) and `performance` (81% negative) are the sharpest pain points.
- **The praise-and-complaint pattern recurs:** ~18% of reviews carry both a positive and a negative aspect, most often gameplay praised against multiplayer or performance criticised, a "want to like it but can't fully" profile.

## Tech stack

`pandas` · `scikit-learn` · `matplotlib` / `seaborn` · `transformers` · `torch` · `datasets` · `BERTopic` (`sentence-transformers`, `umap-learn`, `hdbscan`) · `google-genai` (Gemini) · `nltk`

## Running the notebooks

The notebooks were built for **Google Colab** with data on **Google Drive**.

1. Download the [dataset](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) and place `steam_reviews.csv` at:
   ```
   MyDrive/Colab Notebooks/Projets/steam_reviews/data/raw/steam_reviews.csv
   ```
2. Open each notebook in Colab and run in order (01 to 06). Notebook 02 writes the processed corpus that 03 to 06 consume.
3. Notebook 06 calls the Gemini API: add a `GEMINI_API_KEY` in the **Colab Secrets** panel (🔑) with notebook access enabled. Its free tier is rate-limited (~5 requests/min), so the extraction is paced accordingly.

Dependencies install from the notebooks themselves (`!pip install ...`); `requirements.txt` lists them for a local run.

> **Note:** the raw and processed data files are **not** committed (see `.gitignore`); download the dataset from Kaggle to reproduce.

## Project structure

```
steam-reviews/
├── 01_data_ingestion_eda.ipynb
├── 02_preprocessing_pipeline.ipynb
├── 03_baseline_sentiment_pretrained.ipynb
├── 04_finetuning_benchmark.ipynb
├── 05_topic_modelling_bertopic.ipynb
├── 06_aspect_based_sentiment.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- The aspect-based sample (notebook 06) is **targeted, not representative**: 400 longer reviews, 60% complaints. Comparisons between aspects hold; absolute rates do not describe the corpus.
- The LLM extraction is checked for fabricated quotes but **not validated against human annotation**, so its output is plausible and internally consistent rather than measured against ground truth.

## Author

**Juliette Bouli-Mengue**, Master's in Data Science.
Dataset © Luthfi Mahendra, via Kaggle.
