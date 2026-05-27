# Turkish Legal RAG - Metrics Collected So Far

This file consolidates all metrics/results reported during the project conversation up to the current stage.

Current status: Qwen3-32B Base RAG full run, Qwen3-32B QLoRA training, and Qwen3-32B Fine-tuned RAG full run are completed. The final ablation summary is included at the end of this file.

## 0. Rubric Metric Coverage Snapshot

This section maps the instructor rubric directly to the metrics recorded in this file.

| Rubric item | Status | Metrics / Evidence |
| --- | --- | --- |
| Gold QA benchmark, 150-300 Turkish legal questions | Complete | 190 questions in `data/benchmark/gold_benchmark_v1.csv` |
| Gold Q + A + Doc | Complete | `question`, `gold_answer`, `gold_doc_keys`, `gold_article_keys` |
| Baseline RAG | Complete | Gemma Base RAG and Qwen3-32B Base RAG |
| Embedding tuning ablation | Complete | Qwen3-Embedding-8B LoRA tuned dense retrieval |
| Reranker ablation / tuning | Complete | MiniLM reranker, Qwen3-Reranker-8B, Qwen3-Reranker-8B LoRA v5 |
| LLM fine-tuning | Complete | Gemma LoRA and Qwen3-32B QLoRA |
| Fully optimized system | Complete | Qwen3-Embedding-8B dense top-30 + Qwen3-Reranker-8B top-10 + Qwen3-32B |
| Recall@5 / Recall@10 | Complete | doc/article recall@5 and recall@10 in retrieval tables |
| MRR | Complete | doc_mrr and article_mrr |
| nDCG | Complete | doc/article nDCG@5 and nDCG@10 |
| Exact Match | Complete | exact_match |
| F1 Score | Complete | token_f1 |
| BLEU / ROUGE | Complete | ROUGE-L used as the lexical overlap metric |
| Faithfulness score | Complete | Full LLM judge `faithfulness = 3.789 / 5` |
| Citation accuracy | Complete | citation_present, citation_gold_match, grounded_citation_score |
| Hallucination analysis | Complete | unsupported_or_missing_citation, error_type_auto counts, and LLM judge hallucination_risk |
| Custom data support | Complete | `25_teacher_custom_data_single_notebook_demo.ipynb` supports custom docs and optional custom benchmark metrics |

Final selected system metrics:

| Metric group | Key result |
| --- | --- |
| Retrieval | article_hit@5=0.837, article_hit@10=0.874, article_mrr=0.701, article_ndcg@5=0.731 |
| QA / citation | token_f1=0.145, rouge_l=0.123, citation_gold_match=0.821, grounded_citation_score=0.805 |
| Hallucination | unsupported_or_missing_citation=0.142, LLM judge hallucination_risk=1.795/5 |
| LLM judge | correctness=4.053/5, faithfulness=3.789/5, citation_support=3.468/5 |

## 1. Corpus Normalization

Input:

- `data/raw/legal_documents_curated.csv`

Main outputs:

- `data/processed/legal_main_law_corpus_v3.csv`
- `data/processed/legal_main_law_corpus_v3.jsonl`
- `data/processed/legal_qa_auxiliary_v3.csv`
- `data/processed/legal_rejected_review_v3.csv`
- `reports/normalization_report_v3.json`

Normalization metrics:

| Metric | Value |
| --- | ---: |
| raw_rows | 39,223 |
| law_rows | 34,226 |
| qa_rows | 4,997 |
| valid_law_rows | 24,473 |
| rejected_rows | 9,753 |
| suspicious_rows | 9,753 |
| missing_source_url | 0 |
| missing_article_no | 1 |
| missing_article_body | 3 |
| duplicate_groups | 3,886 |
| canonical_source_missing | 0 |
| category_missing | 28,571 |

Rejected/review quality flags:

| quality_flag | Count |
| --- | ---: |
| duplicate_conflict | 6,474 |
| amendment_fragment | 2,334 |
| duplicate_candidate | 939 |
| table_fragment | 3 |
| metadata_only | 2 |
| empty_or_too_short | 1 |

Main corpus quality flags:

| quality_flag | Count |
| --- | ---: |
| valid_article | 24,136 |
| short_but_valid | 337 |

Main corpus schema/result:

| Output | Shape |
| --- | --- |
| `legal_main_law_corpus_v3.csv` | 24,473 rows x 29 columns |
| `legal_qa_auxiliary_v3.csv` | 4,997 rows x 11 columns |
| `legal_rejected_review_v3.csv` | 9,753 rows x 29 columns |

Important decision:

- Only official law/article records are used as the main retrieval corpus.
- QA records are auxiliary training/style data only.

## 2. Benchmark

Benchmark file:

- `data/benchmark/gold_benchmark_v1.csv`

Benchmark metrics:

| Metric | Value |
| --- | ---: |
| question_count | 190 |
| columns | 16 |
| missing_article_keys | 0 |
| benchmark coverage in normalized corpus | 190 / 190 |

Benchmark columns:

- `question_id`
- `topic`
- `question`
- `gold_answer`
- `gold_doc_keys`
- `gold_article_keys`
- `gold_law`
- `gold_article_no`
- `difficulty`
- `source`
- `source_url`
- `benchmark_tier`
- `question_type`
- `verification_status`
- `review_priority`
- `notes`

## 3. Old Kaggle-only Baseline

Old file:

- `data/benchmark/evaluation_results_baseline_old.csv`

Old baseline result:

| Metric | Value |
| --- | ---: |
| source_hit@5 | ~0.795 |
| article_hit@5 | ~0.053 |
| joint_source_article_hit@5 | ~0.047 |
| mrr_source_article | ~0.035 |
| ndcg_source_article@5 | ~0.038 |

Interpretation:

- Old system often found the correct law/source.
- It almost never found the exact article.
- This justified corpus normalization, article-level records, and stable article IDs.

## 4. Index Build - BGE-M3 Baseline

Index:

- `indexes/official_law_v3`

Model:

- `BAAI/bge-m3`

Index metrics:

| Metric | Value |
| --- | ---: |
| record_count | 24,473 |
| embedding_model | BAAI/bge-m3 |
| embedding_dim | 1,024 |
| embedding_count | 24,473 |
| dense_enabled | True |
| bm25_enabled | True |
| bm25_doc_count | 24,473 |

Paths:

- `indexes/official_law_v3/metadata/metadata.jsonl`
- `indexes/official_law_v3/dense/faiss.index`
- `indexes/official_law_v3/dense/embeddings.npy`
- `indexes/official_law_v3/bm25/bm25.pkl`

## 5. Retrieval Evaluation - BGE-M3 / BM25 / Hybrid

### 5.1 Corrected Dense/BM25/Hybrid Results

| Mode | Questions | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | doc_mrr | article_mrr | doc_ndcg@5 | doc_ndcg@10 | article_ndcg@5 | article_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 190 | 0.884 | 0.900 | 0.674 | 0.732 | 0.764 | 0.520 | 0.793 | 0.799 | 0.552 | 0.572 |
| bm25 | 190 | 0.911 | 0.942 | 0.432 | 0.532 | 0.725 | 0.295 | 0.768 | 0.779 | 0.319 | 0.352 |
| hybrid 0.55/0.45 | 190 | 0.889 | 0.911 | 0.589 | 0.711 | 0.796 | 0.449 | 0.818 | 0.825 | 0.472 | 0.511 |

Interpretation:

- BGE-M3 dense was the strongest old retrieval setup for article-level retrieval.
- BM25 was strong for document/source hit but weak for exact article hit.
- Hybrid did not beat dense on article_hit@5.

### 5.2 BGE-M3 Dense Candidate Recall / Weight Sweep Highlights

Best observed candidate recall for dense top-30:

| Metric | Value |
| --- | ---: |
| article_hit@30 | ~0.795 |

Dense baseline used for later comparison:

| Metric | Value |
| --- | ---: |
| doc_hit@5 | 0.889 |
| doc_hit@10 | 0.921 |
| doc_hit@30 | 0.932 |
| article_hit@5 | 0.674 |
| article_hit@10 | 0.732 |
| article_hit@30 | 0.795 |
| article_mrr | 0.523 |
| article_ndcg@5 | 0.552 |
| article_ndcg@10 | 0.572 |
| article_ndcg@30 | 0.586 |

## 6. MiniLM Reranker Ablation

Reranker:

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Setup:

- dense top-30 candidates
- rerank top-10

Result:

| Setup | Questions | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | article_mrr | article_ndcg@5 | article_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense | 190 | 0.889 | 0.921 | 0.674 | 0.732 | 0.523 | 0.552 | 0.572 |
| dense top30 + MiniLM reranker | 190 | 0.863 | 0.921 | 0.663 | 0.711 | 0.500 | 0.536 | 0.551 |

Interpretation:

- MiniLM reranker did not improve the BGE-M3 dense baseline.
- It remained an ablation, not the selected final retriever.

## 7. Optimized Retrieval v2 Ablation

Setup:

- law-aware filtering
- query expansion
- dense retrieval

Result:

| Setup | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | article_hit@30 | article_mrr | article_ndcg@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_baseline | 0.889 | 0.921 | 0.674 | 0.732 | 0.795 | 0.523 | 0.552 |
| optimized_v2 | 0.874 | 0.884 | 0.626 | 0.700 | 0.763 | 0.483 | 0.509 |

Additional:

| Metric | Value |
| --- | ---: |
| law_filter_rate | 0.616 |

Interpretation:

- Optimized v2 did not beat dense baseline.
- It is useful as an ablation showing that overly aggressive law filtering can hurt article retrieval.

## 8. Qwen3-Embedding-8B Retrieval

Model:

- `Qwen/Qwen3-Embedding-8B`

Index:

- `indexes/official_law_v3_qwen3_embedding_8b`

Result:

| Setup | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | doc_mrr | article_mrr | article_ndcg@5 | article_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense | 0.884 | 0.900 | 0.674 | 0.732 | 0.764 | 0.520 | 0.552 | 0.572 |
| Qwen3-Embedding-8B dense | 0.968 | 0.979 | 0.774 | 0.837 | 0.833 | 0.624 | 0.653 | 0.674 |

Improvement over BGE-M3 dense:

| Metric | Delta |
| --- | ---: |
| article_hit@5 | +0.100 |
| article_hit@10 | +0.105 |
| article_mrr | +0.104 |
| article_ndcg@5 | +0.100 |

Topic highlights:

| Topic | article_hit@5 | article_hit@10 |
| --- | ---: | ---: |
| Anayasa Hukuku | 0.714 | 0.786 |
| Borçlar Hukuku | 0.679 | 0.821 |
| Ceza Hukuku | 0.821 | 0.857 |
| Ceza Muhakemesi Hukuku | 0.786 | 0.821 |
| Medeni Hukuk | 0.852 | 0.852 |
| Ticaret Hukuku | 0.720 | 0.800 |
| İş Hukuku | 0.846 | 0.923 |

Interpretation:

- Qwen3-Embedding-8B clearly beat BGE-M3.
- It became the new heavy retrieval baseline.

## 9. Qwen3-Embedding-8B + Qwen3-Reranker-8B

Embedding:

- `Qwen/Qwen3-Embedding-8B`

Reranker:

- `Qwen/Qwen3-Reranker-8B`

Setup:

- dense top-30 candidates
- rerank to top-10

Result:

| Setup | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | doc_mrr | article_mrr | article_ndcg@5 | article_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense | 0.884 | 0.900 | 0.674 | 0.732 | 0.764 | 0.520 | 0.552 | 0.572 |
| Qwen3-Embedding-8B dense | 0.968 | 0.979 | 0.774 | 0.837 | 0.833 | 0.624 | 0.653 | 0.674 |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 0.968 | 0.979 | 0.837 | 0.874 | 0.869 | 0.701 | 0.731 | 0.743 |

Improvement over BGE-M3 dense:

| Metric | Delta |
| --- | ---: |
| article_hit@5 | +0.163 |
| article_hit@10 | +0.142 |
| article_mrr | +0.181 |
| article_ndcg@5 | +0.179 |

Topic highlights:

| Topic | article_hit@5 | article_hit@10 |
| --- | ---: | ---: |
| Anayasa Hukuku | 0.821 | 0.857 |
| Borçlar Hukuku | 0.750 | 0.821 |
| Ceza Hukuku | 0.857 | 0.857 |
| Ceza Muhakemesi Hukuku | 0.857 | 0.857 |
| Medeni Hukuk | 0.926 | 1.000 |
| Ticaret Hukuku | 0.720 | 0.760 |
| İş Hukuku | 0.923 | 0.962 |

Final selected retrieval stack:

```text
Question
-> Qwen/Qwen3-Embedding-8B dense retrieval, top-30
-> Qwen/Qwen3-Reranker-8B reranking
-> top-10 legal article context
-> LLM generation with citations
```

For generation with Qwen3-32B, the reranked top-10 output is precomputed from notebook 12 and loaded from CSV to avoid GPU OOM.

Precomputed reranker CSV:

- `outputs/retrieval_eval/qwen3_embedding_8b_dense_top30_qwen3_reranker_8b_predictions_v1.csv`

## 10. Fine-tuning Dataset Preparation

### 10.1 QA Auxiliary Fine-tune Data

Input:

- `data/processed/legal_qa_auxiliary_v3.csv`

Leakage threshold:

- 0.88

Result:

| Metric | Value |
| --- | ---: |
| raw_qa_rows | 4,997 |
| kept_records | 4,996 |
| train_records | 4,497 |
| val_records | 499 |
| removed_records | 1 |

Removed example:

| Removed question | Matched benchmark question | Similarity |
| --- | --- | ---: |
| `Anayasa madde 6'ya göre, egemenlik kime aittir` | `Anayasa'ya göre egemenlik kime aittir?` | 0.902 |

Outputs:

- `data/processed/finetune_train.jsonl`
- `data/processed/finetune_val.jsonl`

### 10.2 External Instructor SFT Data

Input:

- `data/external/llm.jsonl`

Result:

| Metric | Value |
| --- | ---: |
| raw_external_records | 13,758 |
| kept_external_records | 13,758 |
| external_train_records | 12,383 |
| external_val_records | 1,375 |
| existing_train_records | 4,497 |
| existing_val_records | 499 |
| combined_train_records | 16,880 |
| combined_val_records | 1,874 |
| removed_external_records | 0 |

Outputs:

- `data/processed/finetune_external_train.jsonl`
- `data/processed/finetune_external_val.jsonl`
- `data/processed/finetune_train_combined.jsonl`
- `data/processed/finetune_val_combined.jsonl`

## 11. Gemma LoRA Fine-tuning

Base model:

- `google/gemma-2-2b-it`

### 11.1 Smoke Training

| Metric | Value |
| --- | --- |
| output_dir | `models/adapters/gemma_2_2b_it_lora_smoke` |
| train_records | 200 |
| val_records | 50 |
| max_length | 1,536 |
| num_train_epochs | 1.0 |
| learning_rate | 0.0002 |
| batch_size | 1 |
| gradient_accumulation_steps | 8 |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| use_4bit | True |
| step 25 training_loss | 0.246 |
| step 25 validation_loss | 0.331 |

### 11.2 Medium Training

| Metric | Value |
| --- | --- |
| output_dir | `models/adapters/gemma_2_2b_it_lora_combined_v1` |
| train_records | 4,000 |
| val_records | 400 |
| max_length | 1,536 |
| num_train_epochs | 1.0 |
| learning_rate | 0.0002 |
| batch_size | 1 |
| gradient_accumulation_steps | 8 |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| use_4bit | True |
| trainable_params | 20,766,720 |
| all_params | 2,635,108,608 |
| trainable_percent | 0.7881% |
| total_steps | 500 |
| runtime | 44m 51s |
| step 200 training_loss | 0.193 |
| step 200 validation_loss | 0.300 |
| step 400 training_loss | 0.132 |
| step 400 validation_loss | 0.264 |

## 12. Gemma Base RAG Generation

LLM:

- `google/gemma-2-2b-it`

Retriever:

- BGE-M3 dense, top-10 context

### 12.1 Gemma Base Smoke Runs

Smoke run 1:

| Metric | Value |
| --- | ---: |
| question_count | 10 |
| exact_match | 0.000 |
| token_f1 | 0.162 |
| rouge_l | 0.143 |
| retrieval_gold_available | 0.400 |
| citation_present | 0.500 |
| citation_gold_match | 0.100 |
| grounded_citation_score | 0.100 |
| unsupported_or_missing_citation | 0.700 |

Smoke run 2:

| Metric | Value |
| --- | ---: |
| question_count | 10 |
| exact_match | 0.000 |
| token_f1 | 0.130 |
| rouge_l | 0.106 |
| retrieval_gold_available | 0.600 |
| citation_present | 0.700 |
| citation_gold_match | 0.200 |
| grounded_citation_score | 0.100 |
| unsupported_or_missing_citation | 0.500 |

### 12.2 Gemma Base Full 190

| Metric | Value |
| --- | ---: |
| question_count | 190 |
| exact_match | 0.000 |
| token_f1 | 0.144 |
| rouge_l | 0.128 |
| retrieval_gold_available | 0.732 |
| citation_present | 0.568 |
| citation_gold_match | 0.337 |
| grounded_citation_score | 0.289 |
| unsupported_or_missing_citation | 0.574 |

Topic-level Gemma Base:

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.174 | 0.156 | 0.679 | 0.500 | 0.143 | 0.107 | 0.679 |
| Borçlar Hukuku | 0.131 | 0.116 | 0.429 | 0.464 | 0.143 | 0.143 | 0.821 |
| Ceza Hukuku | 0.112 | 0.096 | 0.679 | 0.714 | 0.536 | 0.464 | 0.464 |
| Ceza Muhakemesi Hukuku | 0.147 | 0.126 | 0.821 | 0.679 | 0.357 | 0.357 | 0.429 |
| Medeni Hukuk | 0.164 | 0.147 | 0.852 | 0.370 | 0.259 | 0.185 | 0.704 |
| Ticaret Hukuku | 0.098 | 0.092 | 0.800 | 0.440 | 0.360 | 0.280 | 0.640 |
| İş Hukuku | 0.179 | 0.159 | 0.885 | 0.808 | 0.577 | 0.500 | 0.269 |

Gemma Base error counts:

| error_type_auto | Count |
| --- | ---: |
| missing_citation | 58 |
| retrieval_miss | 51 |
| acceptable_automatic | 28 |
| low_answer_overlap | 27 |
| wrong_or_unsupported_citation | 26 |

## 13. Gemma Fine-tuned RAG Generation

LLM:

- `google/gemma-2-2b-it + LoRA`

Adapter:

- `models/adapters/gemma_2_2b_it_lora_combined_v1`

Retriever:

- BGE-M3 dense, top-10 context

### 13.1 Fine-tuned Smoke

| Metric | Value |
| --- | ---: |
| question_count | 10 |
| exact_match | 0.000 |
| token_f1 | 0.146 |
| rouge_l | 0.119 |
| retrieval_gold_available | 0.600 |
| citation_present | 0.900 |
| citation_gold_match | 0.100 |
| grounded_citation_score | 0.100 |
| unsupported_or_missing_citation | 0.500 |

Fine-tuned smoke error counts:

| error_type_auto | Count |
| --- | ---: |
| wrong_or_unsupported_citation | 4 |
| retrieval_miss | 4 |
| missing_citation | 1 |
| low_answer_overlap | 1 |

### 13.2 Fine-tuned Full 190

| Metric | Value |
| --- | ---: |
| question_count | 190 |
| exact_match | 0.000 |
| token_f1 | 0.127 |
| rouge_l | 0.109 |
| retrieval_gold_available | 0.732 |
| citation_present | 0.642 |
| citation_gold_match | 0.179 |
| grounded_citation_score | 0.163 |
| unsupported_or_missing_citation | 0.563 |

Topic-level Gemma Fine-tuned:

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.162 | 0.145 | 0.679 | 0.679 | 0.107 | 0.071 | 0.571 |
| Borçlar Hukuku | 0.127 | 0.114 | 0.429 | 0.750 | 0.179 | 0.179 | 0.750 |
| Ceza Hukuku | 0.097 | 0.080 | 0.679 | 0.571 | 0.250 | 0.214 | 0.607 |
| Ceza Muhakemesi Hukuku | 0.117 | 0.094 | 0.821 | 0.714 | 0.071 | 0.071 | 0.464 |
| Medeni Hukuk | 0.117 | 0.096 | 0.852 | 0.667 | 0.222 | 0.222 | 0.481 |
| Ticaret Hukuku | 0.125 | 0.107 | 0.800 | 0.560 | 0.200 | 0.200 | 0.560 |
| İş Hukuku | 0.142 | 0.129 | 0.885 | 0.538 | 0.231 | 0.192 | 0.500 |

Fine-tuned full error counts:

| error_type_auto | Count |
| --- | ---: |
| missing_citation | 56 |
| wrong_or_unsupported_citation | 52 |
| retrieval_miss | 51 |
| low_answer_overlap | 16 |
| acceptable_automatic | 15 |

Comparison:

| System | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma Base RAG | 0.144 | 0.128 | 0.732 | 0.568 | 0.337 | 0.289 | 0.574 |
| Gemma LoRA RAG | 0.127 | 0.109 | 0.732 | 0.642 | 0.179 | 0.163 | 0.563 |
| Delta FT - Base | -0.017 | -0.018 | 0.000 | +0.074 | -0.158 | -0.126 | -0.011 |

Interpretation:

- Gemma LoRA increased citation presence.
- It reduced citation correctness and grounded citation score.
- Gemma Base remained stronger on citation correctness.

## 14. Experiment Summary Table Reported

| experiment_id | stage | Important values |
| --- | --- | --- |
| corpus_normalization_v3 | corpus | raw_rows=39,223; valid_law_rows=24,473; qa_rows=4,997; rejected_rows=9,753 |
| finetune_dataset_qa_aux | data | train_records=4,497; val_records=499; removed_records=1 |
| finetune_dataset_external_combined | data | train_records=16,880; val_records=1,874; removed_records=0 |
| base_rag_dense_top10 | generation | token_f1=0.144; rouge_l=0.128; grounded_citation_score=0.289 |
| finetuned_rag_lora_medium_dense_top10 | generation | token_f1=0.127; rouge_l=0.109; grounded_citation_score=0.163 |

## 15. External Dataset Audit

External instructor files:

| File | Count |
| --- | ---: |
| `data/external/corpus.jsonl` | 7,579 |
| `data/external/llm.jsonl` | 13,758 |
| `data/external/embedding.jsonl` | 2,059 |
| `data/external/reranker.jsonl` | 6,752 |
| `data/external/gold_benchmark.json` | 240 |
| `data/external/rag_eval.json` | 1,000 |

Usage:

- `llm.jsonl` used for SFT/QLoRA data.
- `embedding.jsonl` and `reranker.jsonl` are available for optional tuning ablations, but these are currently deprioritized.
- External data is not used as the official-law retrieval corpus.


### 17.1 Qwen3-32B Base Smoke - 10 Questions

| Metric | Value |
| --- | ---: |
| question_count | 10 |
| exact_match | 0.000 |
| token_f1 | 0.164 |
| rouge_l | 0.133 |
| retrieval_gold_available | 0.900 |
| citation_present | 1.000 |
| citation_gold_match | 0.900 |
| grounded_citation_score | 0.900 |
| unsupported_or_missing_citation | 0.100 |

Topic-level Qwen3-32B Base Smoke:

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.176 | 0.144 | 0.875 | 1.000 | 0.875 | 0.875 | 0.125 |
| Borçlar Hukuku | 0.117 | 0.088 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Qwen3-32B Base Smoke error counts:

| error_type_auto | Count |
| --- | ---: |
| acceptable_automatic | 6 |
| low_answer_overlap | 3 |
| retrieval_miss | 1 |

Interpretation:

- Smoke result is much stronger than Gemma on citation/grounding metrics.
- Full 190-question run is in progress.

### 17.2 Qwen3-32B Base Full

Status:

- Running.
- No full metrics yet in this consolidated file.

Expected output files:

- `outputs/generation_eval/qwen3_32b_base_rag_predictions_v1.csv`
- `outputs/generation_eval/qwen3_32b_base_rag_eval_v1.csv`
- `outputs/generation_eval/qwen3_32b_base_rag_summary_v1.json`

## 18. Qwen3-32B QLoRA Fine-tuning Plan

Notebook:

- `notebooks/14_qwen3_32b_qlora_finetune.ipynb`

Planned model:

- `Qwen/Qwen3-32B`

Adapter:

- `models/adapters/qwen3_32b_qlora_combined_v1`

Training files:

- `data/processed/finetune_train_combined.jsonl`
- `data/processed/finetune_val_combined.jsonl`

Planned heavy settings:

| Setting | Value |
| --- | --- |
| max_length | 2,048 |
| max_train_samples | None |
| max_val_samples | None |
| num_train_epochs | 1.0 |
| learning_rate | 1e-4 |
| train batch size | 1 |
| eval batch size | 1 |
| gradient_accumulation_steps | 16 |
| LoRA r | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| 4-bit QLoRA | True |

Fallback if OOM:

- `max_length=1536`
- `lora_r=16`
- `lora_alpha=32`

## 19. Qwen3-32B Fine-tuned RAG Plan

Notebook:

- `notebooks/15_qwen3_32b_finetuned_rag_eval.ipynb`

Same comparison constraints:

- Same corpus
- Same benchmark
- Same retrieval
- Same precomputed reranked top-10 context
- Same prompt
- Same decoding
- Same base checkpoint, plus QLoRA adapter

Expected outputs:

- `outputs/generation_eval/qwen3_32b_finetuned_rag_predictions_v1.csv`
- `outputs/generation_eval/qwen3_32b_finetuned_rag_eval_v1.csv`
- `outputs/generation_eval/qwen3_32b_finetuned_rag_summary_v1.json`

## 20. Hallucination / Error Analysis

Implemented deterministic hallucination/error proxies:

- `retrieval_gold_available`
- `citation_present`
- `citation_gold_match`
- `grounded_citation_score`
- `unsupported_or_missing_citation`
- `error_type_auto`

Error types:

- `retrieval_miss`
- `missing_citation`
- `wrong_or_unsupported_citation`
- `low_answer_overlap`
- `acceptable_automatic`

Final notebook:

- `notebooks/20_final_hallucination_error_analysis.ipynb`

Expected report files:

- `reports/error_analysis_gemma_base_rag.csv`
- `reports/error_analysis_gemma_finetuned_rag.csv`
- `reports/error_analysis_qwen3_32b_base_rag.csv`
- `reports/error_analysis_qwen3_32b_finetuned_rag.csv`
- `reports/final_hallucination_error_summary.csv`
- `reports/final_hallucination_error_summary.md`

## 21. Completion Status / Remaining Work

Completed since the earlier project stage:

1. `16_final_best_system_compare.ipynb` was completed and its final comparison tables are recorded below.
2. `20_final_hallucination_error_analysis.ipynb` was completed and its hallucination/error metrics are recorded below.
3. Embedding and reranker fine-tuning ablations were completed and evaluated.
4. Full LLM judge evaluation was completed on all 190 benchmark questions.
5. Custom data support was consolidated into `25_teacher_custom_data_single_notebook_demo.ipynb`.

Remaining delivery work:

- Prepare the final report and presentation.
- Optionally run `25_teacher_custom_data_single_notebook_demo.ipynb` with a teacher-provided custom benchmark if one is supplied.

Final tuning decision:

- Embedding and reranker tuning were implemented, leakage-checked, and evaluated.
- They did not improve the locked official-law benchmark.
- Therefore the final selected retrieval stack remains the stronger measured zero-shot Qwen3-Embedding-8B + Qwen3-Reranker-8B setup.

## 22. Final Ablation Summary

This is the most important ablation table for the report. It shows the improvement path from the old corpus to the final Qwen3 RAG system.

### 22.1 Retrieval Ablation

| Experiment | Corpus | Embedding / Retriever | Reranker | article_hit@5 | article_hit@10 | article_mrr | article_ndcg@5 | Main interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Old Kaggle-only baseline | Kaggle-only | old setup | none | ~0.053 | n/a | ~0.035 | ~0.038 | Correct article retrieval was almost failing. |
| Official corpus + BGE-M3 dense | official-law v3 | BGE-M3 dense | none | 0.674 | 0.732 | 0.520 | 0.552 | Corpus normalization gave the largest early improvement. |
| BM25 only | official-law v3 | BM25 | none | 0.432 | 0.532 | 0.295 | 0.319 | BM25 found sources, but exact article retrieval was weak. |
| Hybrid BGE-M3 + BM25 | official-law v3 | dense + BM25 | none | 0.589 | 0.711 | 0.449 | 0.472 | Hybrid did not beat dense for article-level retrieval. |
| BGE-M3 + MiniLM reranker | official-law v3 | BGE-M3 top-30 | MiniLM | 0.663 | 0.711 | 0.500 | 0.536 | Small reranker did not improve the BGE-M3 dense baseline. |
| Law-aware optimized v2 | official-law v3 | law-aware dense | none | 0.626 | 0.700 | 0.483 | 0.509 | Aggressive law filtering hurt article retrieval. |
| Qwen3-Embedding-8B dense | official-law v3 | Qwen3-Embedding-8B | none | 0.774 | 0.837 | 0.624 | 0.653 | Stronger embedding substantially improved article retrieval. |
| Final retrieval | official-law v3 | Qwen3-Embedding-8B top-30 | Qwen3-Reranker-8B | 0.837 | 0.874 | 0.701 | 0.731 | Best retrieval stack. |

Retrieval improvement from old Kaggle-only to final retrieval:

```text
article_hit@5: ~0.053 -> 0.837
approximate improvement: 15.8x
```

### 22.2 Generation / Same-LLM Fine-tuning Ablation

| System | LLM | Fine-tune | Retrieval stack | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma Base RAG | Gemma-2-2B-it | none | BGE-M3 dense | 0.144 | 0.128 | 0.732 | 0.568 | 0.337 | 0.289 | 0.574 |
| Gemma LoRA RAG | Gemma-2-2B-it | LoRA | BGE-M3 dense | 0.127 | 0.109 | 0.732 | 0.642 | 0.179 | 0.163 | 0.563 |
| Qwen3-32B Base RAG | Qwen3-32B | none | Qwen3-Embedding-8B + Qwen3-Reranker-8B | 0.145 | 0.123 | 0.874 | 0.984 | 0.821 | 0.805 | 0.142 |
| Qwen3-32B QLoRA RAG | Qwen3-32B | QLoRA | Qwen3-Embedding-8B + Qwen3-Reranker-8B | 0.191 | 0.167 | 0.874 | 0.932 | 0.742 | 0.721 | 0.195 |

Same-LLM Qwen3 comparison:

| Metric | Qwen3 Base | Qwen3 QLoRA | Delta |
| --- | ---: | ---: | ---: |
| token_f1 | 0.145 | 0.191 | +0.046 |
| rouge_l | 0.123 | 0.167 | +0.044 |
| retrieval_gold_available | 0.874 | 0.874 | 0.000 |
| citation_present | 0.984 | 0.932 | -0.052 |
| citation_gold_match | 0.821 | 0.742 | -0.079 |
| grounded_citation_score | 0.805 | 0.721 | -0.084 |
| unsupported_or_missing_citation | 0.142 | 0.195 | +0.053 |

Interpretation:

- Qwen3 QLoRA improved answer overlap and answer style quality.
- Qwen3 Base remained stronger on citation precision and grounding.
- Retrieval stayed identical, so the difference is caused by LLM fine-tuning rather than retrieval changes.
- This is a valid ablation result: fine-tuning improved answer quality metrics but introduced a moderate citation-discipline trade-off.

### 22.3 Hallucination / Error Ablation

| System | retrieval_miss | missing_citation | wrong_or_unsupported_citation | low_answer_overlap | acceptable_automatic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma Base RAG | 51 | 58 | 26 | 27 | 28 |
| Gemma LoRA RAG | 51 | 56 | 52 | 16 | 15 |
| Qwen3-32B Base RAG | 24 | 3 | 10 | 78 | 75 |
| Qwen3-32B QLoRA RAG | 24 | 13 | 16 | 42 | 95 |

Interpretation:

- Qwen3 retrieval reduced retrieval misses from 51 to 24.
- Qwen3 Base nearly eliminated missing citations.
- Qwen3 QLoRA reduced low answer overlap and increased acceptable automatic cases.
- Qwen3 QLoRA had more missing/wrong citations than Qwen3 Base, so citation repair or stricter prompting could be future work.

### 22.4 Final Hallucination / Error Analysis Output

This table was produced by `20_final_hallucination_error_analysis.ipynb`.

| system | status | exact_match | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation | error_missing_citation | error_retrieval_miss | error_acceptable_automatic | error_low_answer_overlap | error_wrong_or_unsupported_citation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma_base_rag | ok | 0.000 | 0.143630 | 0.127552 | 0.731579 | 0.568421 | 0.336842 | 0.289474 | 0.573684 | 58 | 51 | 28 | 27 | 26 |
| gemma_finetuned_rag | ok | 0.000 | 0.126803 | 0.109211 | 0.731579 | 0.642105 | 0.178947 | 0.163158 | 0.563158 | 56 | 51 | 15 | 16 | 52 |
| qwen3_32b_base_rag | ok | 0.000 | 0.144925 | 0.123274 | 0.873684 | 0.984211 | 0.821053 | 0.805263 | 0.142105 | 3 | 24 | 75 | 78 | 10 |
| qwen3_32b_finetuned_rag | ok | 0.000 | 0.191439 | 0.167383 | 0.873684 | 0.931579 | 0.742105 | 0.721053 | 0.194737 | 13 | 24 | 95 | 42 | 16 |

Final interpretation:

- Qwen3-32B Base RAG is the safest citation-grounded system.
- Qwen3-32B QLoRA RAG is the strongest answer-quality system by F1, ROUGE-L, and acceptable automatic count.
- QLoRA improved answer style and overlap but moderately reduced citation precision.
- Retrieval misses stayed identical between Qwen3 Base and Qwen3 QLoRA because both use the same retrieval/reranking stack.

## 23. Full LLM Judge Evaluation

This full LLM-as-judge run was produced with `23_optional_llm_judge_eval.ipynb`.

Evaluated system:

- `Qwen3-32B Base RAG`
- `Qwen3-Embedding-8B dense top-30`
- `Qwen3-Reranker-8B top-10`
- 190 benchmark questions

Judge model:

- `Qwen/Qwen3-32B`

Output files:

- `outputs/generation_eval/qwen3_32b_base_rag_llm_judge_sample.csv`
- `outputs/generation_eval/qwen3_32b_base_rag_llm_judge_sample_summary.json`

Note:

- The output filename contains `sample`, but this run was executed with `limit=None`.
- Therefore `question_count=190` and this is a full benchmark LLM judge evaluation.

Overall judge metrics:

| Judge metric | Score |
| --- | ---: |
| correctness | 4.053 / 5 |
| faithfulness | 3.789 / 5 |
| citation_support | 3.468 / 5 |
| hallucination_risk | 1.795 / 5 |

Topic-level judge metrics:

| Topic | correctness | faithfulness | citation_support | hallucination_risk |
| --- | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 4.321 | 4.250 | 3.821 | 1.500 |
| Borçlar Hukuku | 3.929 | 3.714 | 3.464 | 1.893 |
| Ceza Hukuku | 4.036 | 3.679 | 3.286 | 1.821 |
| Ceza Muhakemesi Hukuku | 3.893 | 3.429 | 3.321 | 1.893 |
| Medeni Hukuk | 4.185 | 3.889 | 3.667 | 1.778 |
| Ticaret Hukuku | 3.600 | 3.480 | 3.040 | 2.200 |
| İş Hukuku | 4.385 | 4.077 | 3.654 | 1.500 |

Interpretation:

- The final base Qwen3 RAG system is judged as generally correct (`4.05/5`).
- Faithfulness to retrieved context is strong (`3.79/5`).
- Citation support is positive but remains the main improvement area (`3.47/5`).
- Hallucination risk is low (`1.79/5`), which supports the deterministic hallucination analysis.
- Ticaret Hukuku is the weakest topic by judge score and has the highest hallucination risk.

## 24. Why ROUGE and F1 Are Relatively Low

ROUGE-L and token-F1 are lexical overlap metrics. They reward word-level similarity between the generated answer and the reference answer. In Turkish legal QA, this is strict because the model can give a legally correct answer with different wording, different word order, or a longer explanatory style.

Example:

```text
Gold answer:
Kiraya veren ihtiyaç, yeniden inşa veya yeni malikin ihtiyacı varsa tahliye davası açabilir.

Generated answer:
Türk Borçlar Kanunu'na göre kiraya veren; konut veya işyeri gereksinimi, taşınmazın yeniden inşası ya da yeni malik gereksinimi gibi kanuni sebepler mevcutsa dava yoluyla tahliye isteyebilir.
```

These answers are legally aligned, but their lexical overlap is limited. Therefore, ROUGE/F1 remain modest even when citation and grounding are strong.

This is why the project reports ROUGE/F1 together with:

- retrieval metrics: Recall@5, Recall@10, MRR, nDCG, article_hit@k
- grounding metrics: grounded_citation_score, citation_gold_match
- hallucination metrics: unsupported_or_missing_citation and error categories

In this project, Qwen3-32B Base RAG had modest ROUGE-L (`0.123`) but strong grounded citation score (`0.805`). After QLoRA, ROUGE-L improved to `0.167`, showing that fine-tuning made answers more similar to benchmark answer style, while citation grounding remained reasonably strong at `0.721`.

## 25. Embedding and Reranker Fine-tuning Ablation

This section records the optional extra-credit retrieval tuning experiments. These were implemented and evaluated after the main Qwen3 system was already complete.

### 24.1 Embedding Fine-tuning Data

Input:

- `data/external/embedding.jsonl`

Leakage check:

- Benchmark similarity threshold: `0.88`
- Removed records: `0`

Split:

| Metric | Value |
| --- | ---: |
| raw_records | 2,059 |
| kept_records | 2,059 |
| train_records | 1,854 |
| val_records | 205 |

### 24.2 Qwen3-Embedding-8B LoRA Fine-tuning

Model:

- `Qwen/Qwen3-Embedding-8B`

Output:

- `models/embedding_tuned/qwen3_embedding_8b_legal_lora_v1`
- `indexes/official_law_v3_qwen3_embedding_8b_legal_tuned`

Training:

| Metric | Value |
| --- | ---: |
| train_examples | 1,854 |
| val_examples | 205 |
| batch_size | 1 |
| epochs | 1 |
| max_seq_length | 1,024 |
| lora_r | 32 |
| lora_alpha | 64 |
| lora_dropout | 0.05 |
| trainable_params | 87,293,952 |
| all_params | 7,654,589,440 |
| trainable_percent | 1.140 |
| runtime | 50m 11s |

Dense retrieval result:

| Experiment | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | doc_mrr | article_mrr | article_ndcg@5 | article_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-8B base dense | 0.968 | 0.979 | 0.774 | 0.837 | 0.833 | 0.624 | 0.653 | 0.674 |
| Qwen3-Embedding-8B LoRA tuned dense | 0.963 | 0.968 | 0.742 | 0.816 | 0.818 | 0.619 | 0.639 | 0.663 |

Interpretation:

- Embedding fine-tuning was successfully implemented.
- It did not improve official-law article retrieval.
- The likely reason is training-data mismatch: the external embedding data includes heterogeneous legal QA, decisions, summaries, and hard negatives, while the benchmark evaluates official article-level mevzuat retrieval.

### 24.3 Reranker Fine-tuning Data

Input:

- `data/external/reranker.jsonl`

Leakage check:

- Benchmark similarity threshold: `0.88`
- Removed records: `0`

Split:

| Metric | Value |
| --- | ---: |
| raw_records | 6,752 |
| kept_records | 6,752 |
| train_records | 6,077 |
| val_records | 675 |
| positive_labels | 2,453 |
| negative_labels | 4,299 |

### 24.4 Qwen3-Reranker-8B LoRA Fine-tuning

Model:

- `Qwen/Qwen3-Reranker-8B`

Final clean adapter:

- `models/reranker_tuned/qwen3_reranker_8b_legal_lora_v5`

Important implementation note:

- Earlier `v4` adapter files were produced through an incompatible `CrossEncoder`/`CAUSAL_LM` path and were not used for final evaluation.
- `v5` was trained cleanly with `AutoModelForSequenceClassification`, `BCEWithLogitsLoss`, and `TaskType.SEQ_CLS`.

### 24.5 Retrieval Tuning Results

| Experiment | Embedding | Reranker | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | doc_mrr | article_mrr | article_ndcg@5 | article_ndcg@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final zero-shot retrieval | Qwen3-Embedding-8B base | Qwen3-Reranker-8B base | 0.968 | 0.979 | 0.837 | 0.874 | 0.869 | 0.701 | 0.731 | 0.743 |
| Reranker tuned only | Qwen3-Embedding-8B base | Qwen3-Reranker-8B LoRA v5 | 0.953 | 0.968 | 0.737 | 0.832 | 0.812 | 0.535 | 0.576 | 0.607 |
| Embedding + reranker tuned | Qwen3-Embedding-8B LoRA tuned | Qwen3-Reranker-8B LoRA v5 | 0.926 | 0.958 | 0.705 | 0.821 | 0.779 | 0.525 | 0.558 | 0.596 |

Interpretation:

- Reranker fine-tuning was successfully implemented and evaluated.
- Tuned reranker did not beat the zero-shot Qwen3-Reranker-8B.
- Fully tuned retrieval also did not beat the zero-shot final retrieval stack.
- Final system therefore keeps the empirically best retrieval setup:

```text
Qwen/Qwen3-Embedding-8B base
+ Qwen/Qwen3-Reranker-8B base
```

This is still a valid ablation result: the tuned components were built, leakage-checked, evaluated, and rejected because the measured benchmark performance was lower than the zero-shot Qwen3 retrieval stack.

### 24.6 Base vs Fine-Tuned RAG Semantic Similarity

In addition to exact match, token F1, and ROUGE-L, answer-level semantic similarity was computed between each generated answer and the corresponding gold answer. The metric was computed with `Qwen/Qwen3-Embedding-8B` by embedding `generated_answer` and `gold_answer`, then taking cosine similarity.

| System | mean semantic similarity | median semantic similarity | std |
| --- | ---: | ---: | ---: |
| Base RAG | 0.7890 | 0.8064 | 0.0835 |
| Fine-tuned RAG | 0.7908 | 0.8072 | 0.0911 |

Interpretation:

- Fine-tuned RAG achieved a small positive gain in answer-level semantic similarity.
- The improvement is modest, but it is directionally consistent with the larger gains observed in token F1 and ROUGE-L.
- This indicates that fine-tuning made the generated answers slightly closer to the gold answers at the meaning level, not only at the lexical-overlap level.

Psychological and user-safety interpretation:

- Higher semantic similarity can improve perceived answer relevance because the response is closer to the expected legal explanation.
- However, in legal QA, a more fluent or semantically similar answer may also create stronger user trust. Therefore, semantic similarity must be interpreted together with citation and grounding metrics.
- Since the fine-tuned RAG improved answer overlap but reduced citation-grounding scores, the final system should not be selected only by answer similarity. The safer user-facing system is the one that balances relevance with reliable legal grounding and avoids unsupported confidence.

### 24.7 Leakage-Safe External-Mapped Retrieval Tuning Result

After the initial external-data tuning experiments, a safer external-mapped tuning pipeline was prepared. In this version, the locked final benchmark is not used for tuning. External training examples are mapped to the official-law schema only when their law/article identifiers can be matched to the normalized official corpus.

New embedding adapter:

- `models/embedding_tuned/qwen3_embedding_8b_external_mapped_lora_v1`

New reranker adapter:

- `models/reranker_tuned/qwen3_reranker_8b_external_mapped_lora_v1`

New tuned embedding index:

- `indexes/official_law_v3_qwen3_embedding_8b_external_mapped_lora_v1`

External-mapped retrieval ablation result:

| Experiment | article_hit@5 | article_hit@10 | article_recall@5 | article_recall@10 | article_mrr | article_ndcg@5 | article_ndcg@10 | doc_hit@10 | doc_mrr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| external_mapped_tuned_embedding_dense_only | 0.7421 | 0.7789 | 0.7421 | 0.7789 | 0.5859 | 0.6211 | 0.6336 | 0.9368 | 0.7578 |
| external_mapped_tuned_embedding_base_reranker | 0.1368 | 0.2737 | 0.1368 | 0.2737 | 0.0838 | 0.0830 | 0.1274 | 0.8316 | 0.5024 |
| external_mapped_tuned_embedding_tuned_reranker | 0.3474 | 0.5053 | 0.3474 | 0.5053 | 0.2102 | 0.2272 | 0.2793 | 0.8579 | 0.4609 |

Interpretation:

- The leakage-safe external-mapped embedding and reranker tuning runs completed successfully.
- Dense-only tuned embedding did not improve over the zero-shot Qwen3-Embedding-8B dense baseline.
- Pairing the tuned embedding with the base reranker caused a large ranking drop, suggesting distribution mismatch between the tuned embedding candidate pool and the base reranker.
- The tuned reranker partially recovered this drop, but the fully tuned external-mapped retrieval stack still remained far below the zero-shot Qwen3 embedding + Qwen3 reranker system.
Earlier tuning runs achieved stronger numbers than the stricter external-mapped tuning setup, likely because they used more training examples. However, the mapped setup is methodologically cleaner because it only keeps examples that can be aligned to the official-law article schema and avoids using the locked benchmark for training. Under this stricter setting, fine-tuning did not outperform the zero-shot Qwen3 retrieval stack.
