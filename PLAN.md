# Lab delivery record

Objective: simple modular Colab notebook satisfying preprocessing, BoW, n-grams, all four pretrained embedding families with two classifiers each, and zero/few-shot LLM comparison.

Approved scope: local source in this folder, public repository `saltypal/Text_Analytics_Lab_Evaluation`, dataset publication and Git push, notebook execution by user in Colab. No local installs or pretrained model downloads. User selected Ollama `qwen3:0.6b` and one central notebook configuration cell.

Decisions: 80/20 stratified split after exact-text deduplication; fixed seed; frozen pretrained embeddings; Logistic Regression and Linear SVM; limited static pretrained vocabulary to reduce download cost; raw natural text for pretrained models; configurable training-only few-shot demonstrations; identical ten-paper comparison subset. No test-driven hyperparameter search.

Latest requested layout: explicit Questions A–E; one dataset preview cell; A preprocessing/BoW split; B training/probability-generation split; C all embeddings/all classifiers/comparison split; D installer/loading function + directly editable system prompt, separate zero-shot and few-shot cells, two training examples per class by default; E styled metric tables, confusion matrices, detailed misclassification analysis. BoW remains a representation demonstration; the current notebook trains the eight required embedding classifiers.

Acceptance: structurally valid notebook; first code cell fetches latest source, second installs dependencies, third configures everything; explicit preprocessing displays; eight embedding-classifier pairs; actual local lightweight execution using existing packages; no fabricated model results. Heavy pretrained and actual Ollama execution remain for Colab.

Local validation evidence: see `experiments/local_validation/runs/` when checks complete. Colab results are generated at execution and downloaded as a ZIP, not claimed to exist beforehand.

Completed validation: run `20261029T141514Z` passed notebook schema and all 18 code-cell compilation checks; real CSV preprocessing/deduplication/disjoint split/BoW; both actual BoW classifiers (accuracy and macro F1 1.000 on this dataset split); all three n-gram normalization/sequence/generation checks; mocked gzip binary/text embedding readers and cache; mocked Ollama requests. No local package installs, model downloads or actual LLM inference. Independent static review found no consequential issues in the default Colab setup. Heavy-model results remain unmeasured until Colab execution.
