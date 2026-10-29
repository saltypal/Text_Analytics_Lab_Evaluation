# AI-powered Research Paper Discovery Assistant

[Open in Google Colab](https://colab.research.google.com/github/saltypal/Text_Analytics_Lab_Evaluation/blob/main/lab.ipynb)

Use **Runtime → Change runtime type → T4 GPU**, then run the notebook top to bottom.

1. Code cell 1 clones the latest public `main` commit into a fresh disposable runtime directory. Rerun it after GitHub source updates; imports are refreshed. The printed commit identifies the code actually used.
2. Code cell 2 installs Python dependencies. Question D installs and loads Ollama through `install_ollama_qwen()`. Nothing needs installing on the local Windows computer.
3. Code cell 3 contains all adjustable experiment settings, including seed, split, embedding vocabulary limit, BERT batch/length/model, classifier settings and Ollama parameters.

## Included work

- Title + abstract preprocessing: tokenization, case folding, stopword removal, Porter stemming, with intermediate tables.
- Training-fitted sparse Bag of Words and vocabulary counts.
- Unigram, bigram and trigram models, Laplace-smoothed sequence/conditional probabilities, observed-count sample generation.
- Pretrained Word2Vec, GloVe, FastText and frozen BERT embeddings.
- **Logistic Regression and Linear SVM trained separately on each embedding: eight systems.** One cell generates all embeddings; the next trains all eight classifiers; the following cell compares their results.
- **Ollama `qwen3:0.6b`**, thinking disabled, structured JSON labels: zero-shot and few-shot classification in separate cells on the same 10 test papers. Two training demonstrations per class by default. Edit `SYSTEM_PROMPT` and inspect/choose `FEW_SHOT_PAPERS` directly in Question D.
- Accuracy and macro precision, recall and F1; full-test ML results plus a fair paired 10-paper comparison including LLMs.
- Styled comparison tables, confusion matrices, and misclassified titles/abstracts with a summary of domain confusions.

## Notebook presentation order

After fetch/packages/config/setup, one cell previews the dataset. **A:** one preprocessing cell, one BoW cell. **B:** one n-gram construction cell, one probability/text-generation cell. **C:** one all-embedding generation cell, one all-classifier training cell, one comparison cell. **D:** one Ollama installation/loading function and editable system prompt cell, one zero-shot cell, one few-shot cell showing explicit training examples. **E:** a final metric comparison cell, confusion matrix cell, and paper-level error analysis cell. The final cell packages/downloads evidence.

Zero-shot sends the system instructions and the single unseen paper being classified; no example papers or true labels. Few-shot adds only labeled training papers. Both evaluate exactly the same ten held-out papers.

## Small module layout

```text
lab.ipynb                         Colab entry point and all adjustable settings
requirements.txt                  Colab Python dependencies
Q4_Research_Paper_Dataset_1000.csv  Provided dataset
research_assistant/
  preprocessing.py                Input checks, deduplication, text steps, split, BoW
  ngrams.py                       N-gram training, probabilities, generation
  embeddings.py                   Pretrained streaming/pooling, cached BERT features
  classification.py               Two classifiers, metrics, confusion/error tables
  llm.py                          Ollama setup, prompts, ten-paper predictions
experiments/
  local_validation/               Local checks and their run evidence
  lab_evaluation/runs/<run-id>/    Generated Colab evaluation evidence
```

The notebook is explicitly placed at the project root as requested. No second notebook source is maintained.

## Runtime/download choices

The static embedding loader reads only the first **50,000 pretrained entries** by default, retains dataset words and closes the network stream early. These are real pretrained vectors, never randomly initialized substitutes. Raising `embedding_vocab_limit` improves available vocabulary at greater network cost. Coverage and zero-vector document counts are printed. The default is a limited-vocabulary experiment, not an evaluation of the full pretrained vocabulary.

Static embeddings use lowercase natural word tokens, not Porter stems, because stems would lose many pretrained matches. BERT uses the natural title/abstract and attention-mask mean pooling. FastText uses an exported word-vector table; OOV subword synthesis is unavailable without the much larger full model. Gensim is not required: small NumPy parsers avoid SciPy/Gensim compatibility problems.

BERT and Qwen still need model downloads in Colab. Ollama is installed using its official installer, started locally inside the Colab runtime, then pulls the model. The code does not expose an external server or require an API key.

## Evaluation limits

The supplied CSV has 1,000 rows, 200 per class. Eleven exact title/abstract duplicates are removed before splitting. The fixed split has 791 training and 198 test rows. Related template-like papers may still cross the split; scores do not establish performance on arbitrary real papers. Classifiers have fixed hyperparameters; test results are not used for tuning.

Few-shot demonstrations come exclusively from training. The 10-paper LLM test sample contains two papers per class and is reused for the paired ML comparison. Full-test and ten-paper results are separate; ten examples do not support a strong model-ranking claim. Invalid LLM output is retained as `INVALID` and scored as an error. A download/server failure raises an error instead of producing invented metrics.

## Outputs and recovery

Colab writes run artifacts under `/content/experiments/lab_evaluation/runs/<run-id>/`, outside disposable code clones. Each run includes config/environment/source commit, dataset checksum, split manifests, incremental Python stdout/stderr and cell status, metric CSVs, predictions, prompts/raw LLM responses, embedding matrices and confusion/error figures. Model caches are under `/content/research_lab_cache/` plus Hugging Face/Ollama managed caches.

The final notebook cell downloads a ZIP. **Also save the executed `.ipynb`** via Colab's File → Download to preserve rich outputs and installation/native process messages. Colab storage is temporary; neither the cache nor the run directory survives a runtime reset unless downloaded or saved elsewhere. The results ZIP is an evidence bundle, not a full training-resume checkpoint.

To update imported code, edit the local modules, commit and push, then rerun cell 1 and subsequent cells. A notebook-cell change requires replacing that cell or reopening the latest GitHub notebook; cloning code cannot alter cells in an already open notebook.

## Validation performed

Local run `20261029T141514Z`: notebook schema and all 18 code-cell compilation checks passed. Real dataset preprocessing, split, BoW and both baseline classifiers executed; both achieved accuracy/macro-F1 1.000 on the 198-paper test split. N-gram normalization, sequence probability and generation checks passed. Embedding gzip readers/cache and Ollama payloads passed **mocked** tests. Actual pretrained embedding inference and Ollama/Qwen execution were not run locally and remain Colab checks. No local package installation was performed. Full local evidence is under `experiments/local_validation/runs/20261029T141514Z/` (excluded from Git).

## Primary documentation

- [Pretrained vector catalog and provenance](https://github.com/piskvorky/gensim-data)
- [BERT base uncased model](https://huggingface.co/google-bert/bert-base-uncased)
- [Ollama Qwen3 0.6B](https://ollama.com/library/qwen3:0.6b)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama thinking setting](https://docs.ollama.com/capabilities/thinking)
