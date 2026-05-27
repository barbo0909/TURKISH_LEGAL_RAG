# External-Mapped Fine-Tuning Colab Guide

This is the leakage-free tuning flow. It does not use the locked `data/benchmark/gold_benchmark_v1.csv` questions for training. Instead, it maps external training examples to the final official-law schema when `law_no` and `article_no` can be matched to `data/processed/legal_main_law_corpus_v3.csv`.

## Run Order

1. `notebooks/26_external_mapped_prepare_tuning_data_colab.ipynb`
2. `notebooks/27_external_mapped_embedding_lora_colab.ipynb`
3. `notebooks/28_external_mapped_reranker_lora_colab.ipynb`
4. `notebooks/29_external_mapped_embedding_reranker_ablation_colab.ipynb`
5. `notebooks/30_external_mapped_llm_qlora_colab.ipynb`
6. `notebooks/31_external_mapped_full_system_eval_colab.ipynb`

## Why This Replaces the Previous Official-Grounded Flow

The previous `official_grounded` notebooks used the locked 190-question benchmark to create tuning data. That is useful for debugging, but it is not valid as a final held-out comparison. This external-mapped flow keeps the final benchmark untouched and uses it only for evaluation.
