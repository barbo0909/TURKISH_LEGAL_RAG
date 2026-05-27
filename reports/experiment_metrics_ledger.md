# Turkish Legal RAG - Complete Experiment Metrics Ledger

This ledger is intended for the final report. It lists every major experiment, what was used, which metrics were measured, and what the result was.

Status note: Qwen3-32B Base RAG full run, Qwen3-32B QLoRA training, and Qwen3-32B QLoRA RAG full evaluation are completed.

## 0. Required Metrics Coverage

| Required item | Status | Where recorded |
| --- | --- | --- |
| 150-300 Turkish legal gold QA benchmark | Complete | Section B4 |
| Gold question-answer-document setup | Complete | Section B4 |
| Baseline RAG | Complete | E6, E10 |
| Embedding tuning | Complete | E12, Section G4 |
| Reranker / reranker tuning | Complete | E9, E13, Section G5 |
| LLM fine-tuning | Complete | E7, E11, Section G3 |
| Fully optimized system | Complete | E9 + E10 final selected stack |
| Recall@5 / Recall@10 | Complete | Section C |
| MRR | Complete | Section C |
| nDCG | Complete | Section C |
| Exact Match | Complete | Section D |
| F1 Score | Complete | Section D |
| BLEU / ROUGE | Complete | ROUGE-L in Section D |
| Faithfulness score | Complete | Section E4, LLM judge faithfulness |
| Citation accuracy | Complete | Section D and Section E |
| Hallucination analysis | Complete | Section E, deterministic errors + LLM judge hallucination risk |
| Custom data / teacher collection support | Complete | `25_teacher_custom_data_single_notebook_demo.ipynb` |

Final selected system:

```text
official-law v3 corpus
Qwen/Qwen3-Embedding-8B dense top-30
Qwen/Qwen3-Reranker-8B top-10
Qwen/Qwen3-32B Base generation
```

Final selected system headline metrics:

| Metric group | Metrics |
| --- | --- |
| Retrieval | article_hit@5=0.837, article_hit@10=0.874, article_mrr=0.701, article_ndcg@5=0.731 |
| Generation | exact_match=0.000, token_f1=0.145, rouge_l=0.123 |
| Citation / grounding | citation_present=0.984, citation_gold_match=0.821, grounded_citation_score=0.805 |
| Hallucination | unsupported_or_missing_citation=0.142, missing_citation=3, wrong_or_unsupported_citation=10 |
| LLM judge | correctness=4.053/5, faithfulness=3.789/5, citation_support=3.468/5, hallucination_risk=1.795/5 |

## A. Experiment Timeline

| ID | Experiment | Corpus | Retriever | Reranker | LLM | Fine-tune | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Kaggle-only old baseline | Kaggle-only old corpus | old retrieval setup | none | old setup | none | completed |
| E1 | Official corpus + BGE-M3 dense | official law corpus v3 | BGE-M3 dense | none | none | none | completed |
| E2 | BM25 ablation | official law corpus v3 | BM25 | none | none | none | completed |
| E3 | Hybrid retrieval ablation | official law corpus v3 | BGE-M3 + BM25 | none | none | none | completed |
| E4 | MiniLM reranker ablation | official law corpus v3 | BGE-M3 dense top-30 | MiniLM cross-encoder | none | none | completed |
| E5 | Optimized law-aware retrieval v2 | official law corpus v3 | law-aware dense | none | none | none | completed |
| E6 | Gemma Base RAG | official law corpus v3 | BGE-M3 dense | none | Gemma-2-2B-it | none | completed |
| E7 | Gemma LoRA RAG | official law corpus v3 | BGE-M3 dense | none | Gemma-2-2B-it | LoRA/QLoRA | completed |
| E8 | Qwen3 embedding ablation | official law corpus v3 | Qwen3-Embedding-8B dense | none | none | none | completed |
| E9 | Qwen3 reranker ablation | official law corpus v3 | Qwen3-Embedding-8B dense top-30 | Qwen3-Reranker-8B | none | none | completed |
| E10 | Qwen3-32B Base RAG | official law corpus v3 | Qwen3-Embedding-8B dense top-30 | Qwen3-Reranker-8B top-10, precomputed | Qwen3-32B | none | completed |
| E11 | Qwen3-32B QLoRA RAG | official law corpus v3 | Qwen3-Embedding-8B dense top-30 | Qwen3-Reranker-8B top-10, precomputed | Qwen3-32B | QLoRA | completed |
| E12 | Qwen3 embedding tuning | official law corpus v3 | Qwen3-Embedding-8B LoRA tuned | none | none | embedding LoRA | completed |
| E13 | Qwen3 reranker tuning | official law corpus v3 | Qwen3-Embedding-8B dense top-30 | Qwen3-Reranker-8B LoRA v5 | none | reranker LoRA | completed |
| E14 | Fully tuned retrieval | official law corpus v3 | Qwen3-Embedding-8B LoRA tuned top-30 | Qwen3-Reranker-8B LoRA v5 | none | embedding + reranker LoRA | completed |
| E15 | Full LLM judge evaluation | official law corpus v3 | Qwen3-Embedding-8B dense top-30 | Qwen3-Reranker-8B top-10, precomputed | Qwen3-32B Base RAG output judged by Qwen3-32B | none | completed |

## B. Corpus and Benchmark Metrics

### B1. Normalization

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

### B2. Main Corpus Quality Flags

| quality_flag | Count |
| --- | ---: |
| valid_article | 24,136 |
| short_but_valid | 337 |

### B3. Rejected / Review Quality Flags

| quality_flag | Count |
| --- | ---: |
| duplicate_conflict | 6,474 |
| amendment_fragment | 2,334 |
| duplicate_candidate | 939 |
| table_fragment | 3 |
| metadata_only | 2 |
| empty_or_too_short | 1 |

### B4. Benchmark

| Metric | Value |
| --- | ---: |
| benchmark_questions | 190 |
| benchmark_columns | 16 |
| missing_article_keys | 0 |
| gold_article_coverage_in_corpus | 190 / 190 |

## C. Retrieval Metrics

Retrieval metrics include:

- `doc_hit@5`
- `doc_hit@10`
- `article_hit@5`
- `article_hit@10`
- `doc_recall@5`
- `doc_recall@10`
- `article_recall@5`
- `article_recall@10`
- `doc_mrr`
- `article_mrr`
- `doc_ndcg@5`
- `doc_ndcg@10`
- `article_ndcg@5`
- `article_ndcg@10`

### C1. Old Kaggle-only Baseline

| Metric | Value |
| --- | ---: |
| source_hit@5 | ~0.795 |
| article_hit@5 | ~0.053 |
| joint_source_article_hit@5 | ~0.047 |
| mrr_source_article | ~0.035 |
| ndcg_source_article@5 | ~0.038 |

### C2. Main Retrieval Experiments

| Experiment | doc_hit@5 | doc_hit@10 | article_hit@5 | article_hit@10 | doc_recall@5 | doc_recall@10 | article_recall@5 | article_recall@10 | doc_mrr | article_mrr | doc_ndcg@5 | doc_ndcg@10 | article_ndcg@5 | article_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 BGE-M3 dense | 0.884 | 0.900 | 0.674 | 0.732 | 0.884 | 0.900 | 0.674 | 0.732 | 0.764 | 0.520 | 0.793 | 0.799 | 0.552 | 0.572 |
| E2 BM25 | 0.911 | 0.942 | 0.432 | 0.532 | 0.911 | 0.942 | 0.432 | 0.532 | 0.725 | 0.295 | 0.768 | 0.779 | 0.319 | 0.352 |
| E3 Hybrid 0.55 dense / 0.45 BM25 | 0.889 | 0.911 | 0.589 | 0.711 | 0.889 | 0.911 | 0.589 | 0.711 | 0.796 | 0.449 | 0.818 | 0.825 | 0.472 | 0.511 |
| E4 BGE-M3 dense top30 + MiniLM reranker | 0.863 | 0.921 | 0.663 | 0.711 | 0.863 | 0.921 | 0.663 | 0.711 | 0.721 | 0.500 | 0.751 | 0.771 | 0.536 | 0.551 |
| E5 Optimized law-aware dense v2 | 0.874 | 0.884 | 0.626 | 0.700 | 0.874 | 0.884 | 0.626 | 0.700 | 0.810 | 0.483 | 0.824 | 0.828 | 0.509 | 0.532 |
| E8 Qwen3-Embedding-8B dense | 0.968 | 0.979 | 0.774 | 0.837 | 0.968 | 0.979 | 0.774 | 0.837 | 0.833 | 0.624 | 0.866 | 0.869 | 0.653 | 0.674 |
| E9 Qwen3-Embedding-8B + Qwen3-Reranker-8B | 0.968 | 0.979 | 0.837 | 0.874 | 0.968 | 0.979 | 0.837 | 0.874 | 0.869 | 0.701 | 0.894 | 0.897 | 0.731 | 0.743 |
| E12 Qwen3-Embedding-8B LoRA tuned dense | 0.963 | 0.968 | 0.742 | 0.816 | 0.963 | 0.968 | 0.742 | 0.816 | 0.818 | 0.619 | 0.854 | 0.855 | 0.639 | 0.663 |
| E13 Base embedding + Qwen3-Reranker-8B LoRA v5 | 0.953 | 0.968 | 0.737 | 0.832 | 0.953 | 0.968 | 0.737 | 0.832 | 0.812 | 0.535 | 0.846 | 0.851 | 0.576 | 0.607 |
| E14 Tuned embedding + Qwen3-Reranker-8B LoRA v5 | 0.926 | 0.958 | 0.705 | 0.821 | 0.926 | 0.958 | 0.705 | 0.821 | 0.779 | 0.525 | 0.813 | 0.824 | 0.558 | 0.596 |

### C3. Additional Retrieval Values

| Experiment | article_hit@30 | Notes |
| --- | ---: | --- |
| BGE-M3 dense candidate recall | ~0.795 | dense top-30 candidate recall |
| Optimized v2 | 0.763 | law_filter_rate=0.616 |

### C4. Retrieval Improvement Chain

| Step | article_hit@5 | Relative note |
| --- | ---: | --- |
| Old Kaggle-only | ~0.053 | very weak article retrieval |
| Official corpus + BGE-M3 dense | 0.674 | corpus normalization gave the largest early jump |
| Qwen3-Embedding-8B dense | 0.774 | stronger multilingual embedding |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 0.837 | final selected retrieval stack |
| Qwen3-Embedding-8B LoRA tuned dense | 0.742 | embedding tuning did not improve official article retrieval |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B LoRA v5 | 0.737 | reranker tuning did not improve over zero-shot Qwen3 reranker |
| Qwen3-Embedding-8B LoRA tuned + Qwen3-Reranker-8B LoRA v5 | 0.705 | fully tuned retrieval stack did not beat the zero-shot final stack |

Improvement from old Kaggle-only to final retrieval stack:

```text
article_hit@5: ~0.053 -> 0.837
approximate improvement: 15.8x
```

## D. Generation / QA Metrics

Generation metrics include:

- `exact_match`
- `token_f1`
- `rouge_l`
- `retrieval_gold_available`
- `citation_present`
- `citation_gold_match`
- `grounded_citation_score`
- `unsupported_or_missing_citation`

### D1. Main Generation Experiments

| Experiment | Questions | Retriever | Reranker | LLM | Fine-tune | exact_match | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E6 Gemma Base RAG | 190 | BGE-M3 dense | none | Gemma-2-2B-it | none | 0.000 | 0.144 | 0.128 | 0.732 | 0.568 | 0.337 | 0.289 | 0.574 |
| E7 Gemma LoRA RAG | 190 | BGE-M3 dense | none | Gemma-2-2B-it | LoRA | 0.000 | 0.127 | 0.109 | 0.732 | 0.642 | 0.179 | 0.163 | 0.563 |
| E10 Qwen3-32B Base RAG smoke | 10 | Qwen3-Embedding-8B | Qwen3-Reranker-8B, precomputed | Qwen3-32B | none | 0.000 | 0.164 | 0.133 | 0.900 | 1.000 | 0.900 | 0.900 | 0.100 |
| E10 Qwen3-32B Base RAG full | 190 | Qwen3-Embedding-8B | Qwen3-Reranker-8B, precomputed | Qwen3-32B | none | 0.000 | 0.145 | 0.123 | 0.874 | 0.984 | 0.821 | 0.805 | 0.142 |
| E11 Qwen3-32B QLoRA RAG | 190 | Qwen3-Embedding-8B | Qwen3-Reranker-8B, precomputed | Qwen3-32B | QLoRA | 0.000 | 0.191 | 0.167 | 0.874 | 0.932 | 0.742 | 0.721 | 0.195 |

### D2. Gemma Base RAG Topic Metrics

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.174 | 0.156 | 0.679 | 0.500 | 0.143 | 0.107 | 0.679 |
| Borçlar Hukuku | 0.131 | 0.116 | 0.429 | 0.464 | 0.143 | 0.143 | 0.821 |
| Ceza Hukuku | 0.112 | 0.096 | 0.679 | 0.714 | 0.536 | 0.464 | 0.464 |
| Ceza Muhakemesi Hukuku | 0.147 | 0.126 | 0.821 | 0.679 | 0.357 | 0.357 | 0.429 |
| Medeni Hukuk | 0.164 | 0.147 | 0.852 | 0.370 | 0.259 | 0.185 | 0.704 |
| Ticaret Hukuku | 0.098 | 0.092 | 0.800 | 0.440 | 0.360 | 0.280 | 0.640 |
| İş Hukuku | 0.179 | 0.159 | 0.885 | 0.808 | 0.577 | 0.500 | 0.269 |

### D3. Gemma LoRA RAG Topic Metrics

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.162 | 0.145 | 0.679 | 0.679 | 0.107 | 0.071 | 0.571 |
| Borçlar Hukuku | 0.127 | 0.114 | 0.429 | 0.750 | 0.179 | 0.179 | 0.750 |
| Ceza Hukuku | 0.097 | 0.080 | 0.679 | 0.571 | 0.250 | 0.214 | 0.607 |
| Ceza Muhakemesi Hukuku | 0.117 | 0.094 | 0.821 | 0.714 | 0.071 | 0.071 | 0.464 |
| Medeni Hukuk | 0.117 | 0.096 | 0.852 | 0.667 | 0.222 | 0.222 | 0.481 |
| Ticaret Hukuku | 0.125 | 0.107 | 0.800 | 0.560 | 0.200 | 0.200 | 0.560 |
| İş Hukuku | 0.142 | 0.129 | 0.885 | 0.538 | 0.231 | 0.192 | 0.500 |

### D4. Qwen3-32B Base RAG Smoke Topic Metrics

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.176 | 0.144 | 0.875 | 1.000 | 0.875 | 0.875 | 0.125 |
| Borçlar Hukuku | 0.117 | 0.088 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

### D5. Qwen3-32B Base RAG Full Topic Metrics

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.170 | 0.148 | 0.857 | 0.964 | 0.821 | 0.821 | 0.179 |
| Borclar Hukuku | 0.126 | 0.107 | 0.821 | 0.964 | 0.786 | 0.786 | 0.214 |
| Ceza Hukuku | 0.141 | 0.116 | 0.857 | 1.000 | 0.821 | 0.786 | 0.143 |
| Ceza Muhakemesi Hukuku | 0.151 | 0.130 | 0.857 | 0.964 | 0.750 | 0.750 | 0.179 |
| Medeni Hukuk | 0.160 | 0.137 | 1.000 | 1.000 | 0.852 | 0.852 | 0.000 |
| Ticaret Hukuku | 0.114 | 0.095 | 0.760 | 1.000 | 0.720 | 0.680 | 0.240 |
| Is Hukuku | 0.150 | 0.126 | 0.962 | 1.000 | 1.000 | 0.962 | 0.038 |

### D6. Qwen3-32B QLoRA RAG Full Topic Metrics

| Topic | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 0.211 | 0.193 | 0.857 | 0.964 | 0.786 | 0.786 | 0.179 |
| Borclar Hukuku | 0.167 | 0.147 | 0.821 | 0.929 | 0.679 | 0.679 | 0.250 |
| Ceza Hukuku | 0.178 | 0.156 | 0.857 | 0.929 | 0.786 | 0.714 | 0.214 |
| Ceza Muhakemesi Hukuku | 0.190 | 0.162 | 0.857 | 0.929 | 0.750 | 0.750 | 0.214 |
| Medeni Hukuk | 0.214 | 0.185 | 1.000 | 1.000 | 0.852 | 0.852 | 0.000 |
| Ticaret Hukuku | 0.157 | 0.138 | 0.760 | 0.960 | 0.600 | 0.560 | 0.280 |
| Is Hukuku | 0.222 | 0.191 | 0.962 | 0.808 | 0.731 | 0.692 | 0.231 |

## E. Hallucination / Error Analysis Metrics

Error categories:

- `retrieval_miss`
- `missing_citation`
- `wrong_or_unsupported_citation`
- `low_answer_overlap`
- `acceptable_automatic`

### E1. Error Counts

| Experiment | retrieval_miss | missing_citation | wrong_or_unsupported_citation | low_answer_overlap | acceptable_automatic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma Base RAG full | 51 | 58 | 26 | 27 | 28 |
| Gemma LoRA RAG full | 51 | 56 | 52 | 16 | 15 |
| Qwen3-32B Base RAG smoke | 1 | 0 | 0 | 3 | 6 |
| Qwen3-32B Base RAG full | 24 | 3 | 10 | 78 | 75 |
| Qwen3-32B QLoRA RAG full | 24 | 13 | 16 | 42 | 95 |

### E2. Hallucination/Citation Interpretation

| System | Main observation |
| --- | --- |
| Gemma Base | many missing citations and retrieval misses |
| Gemma LoRA | citation_present improved but wrong_or_unsupported_citation increased |
| Qwen3-32B Base smoke | no missing or wrong citation in first 10; one retrieval miss |
| Qwen3-32B Base full | citation_present and grounded citation accuracy are very strong; remaining errors are mostly low lexical overlap and retrieval misses |
| Qwen3-32B QLoRA full | answer lexical quality improved substantially; citation/grounding declined moderately but remained strong |

### E3. Final Hallucination / Error Analysis Output

| system | status | exact_match | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation | error_missing_citation | error_retrieval_miss | error_acceptable_automatic | error_low_answer_overlap | error_wrong_or_unsupported_citation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma_base_rag | ok | 0.000 | 0.143630 | 0.127552 | 0.731579 | 0.568421 | 0.336842 | 0.289474 | 0.573684 | 58 | 51 | 28 | 27 | 26 |
| gemma_finetuned_rag | ok | 0.000 | 0.126803 | 0.109211 | 0.731579 | 0.642105 | 0.178947 | 0.163158 | 0.563158 | 56 | 51 | 15 | 16 | 52 |
| qwen3_32b_base_rag | ok | 0.000 | 0.144925 | 0.123274 | 0.873684 | 0.984211 | 0.821053 | 0.805263 | 0.142105 | 3 | 24 | 75 | 78 | 10 |
| qwen3_32b_finetuned_rag | ok | 0.000 | 0.191439 | 0.167383 | 0.873684 | 0.931579 | 0.742105 | 0.721053 | 0.194737 | 13 | 24 | 95 | 42 | 16 |

### E4. Full LLM Judge Evaluation

This full LLM-as-judge run was produced with `23_optional_llm_judge_eval.ipynb`.

| Field | Value |
| --- | --- |
| judged_system | Qwen3-32B Base RAG + best retrieval |
| judge_model | Qwen/Qwen3-32B |
| question_count | 190 |
| limit | None |
| predictions_csv | `outputs/generation_eval/qwen3_32b_base_rag_predictions_v1.csv` |
| output_csv | `outputs/generation_eval/qwen3_32b_base_rag_llm_judge_sample.csv` |
| output_summary_json | `outputs/generation_eval/qwen3_32b_base_rag_llm_judge_sample_summary.json` |

The output filename contains `sample`, but this run was executed with `limit=None`; therefore it is a full 190-question judge evaluation.

Overall judge metrics:

| Judge metric | Score |
| --- | ---: |
| judge_correctness | 4.053 / 5 |
| judge_faithfulness | 3.789 / 5 |
| judge_citation_support | 3.468 / 5 |
| judge_hallucination_risk | 1.795 / 5 |

Topic-level judge metrics:

| Topic | judge_correctness | judge_faithfulness | judge_citation_support | judge_hallucination_risk |
| --- | ---: | ---: | ---: | ---: |
| Anayasa Hukuku | 4.321 | 4.250 | 3.821 | 1.500 |
| Borçlar Hukuku | 3.929 | 3.714 | 3.464 | 1.893 |
| Ceza Hukuku | 4.036 | 3.679 | 3.286 | 1.821 |
| Ceza Muhakemesi Hukuku | 3.893 | 3.429 | 3.321 | 1.893 |
| Medeni Hukuk | 4.185 | 3.889 | 3.667 | 1.778 |
| Ticaret Hukuku | 3.600 | 3.480 | 3.040 | 2.200 |
| İş Hukuku | 4.385 | 4.077 | 3.654 | 1.500 |

Interpretation:

- The final base Qwen3 RAG system has strong judged correctness (`4.05/5`).
- Faithfulness is strong (`3.79/5`), supporting the retrieval-grounded setup.
- Citation support is good but remains weaker than general correctness (`3.47/5`).
- Hallucination risk is low (`1.79/5`), which supports the deterministic hallucination/error analysis.
- Ticaret Hukuku remains the weakest topic by LLM judge metrics.

## F. Fine-tuning Data Metrics

### F1. QA Auxiliary Fine-tuning Data

| Metric | Value |
| --- | ---: |
| raw_qa_rows | 4,997 |
| kept_records | 4,996 |
| train_records | 4,497 |
| val_records | 499 |
| removed_records | 1 |
| leakage_threshold | 0.88 |

Removed leakage example:

| Removed question | Matched benchmark question | Similarity |
| --- | --- | ---: |
| Anayasa madde 6'ya göre, egemenlik kime aittir | Anayasa'ya göre egemenlik kime aittir? | 0.902 |

### F2. External Instructor SFT Data

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

## G. Fine-tuning Run Metrics

### G1. Gemma LoRA Smoke

| Metric | Value |
| --- | --- |
| base_model | google/gemma-2-2b-it |
| output_dir | models/adapters/gemma_2_2b_it_lora_smoke |
| train_records | 200 |
| val_records | 50 |
| max_length | 1,536 |
| max_train_samples | 200 |
| max_val_samples | 50 |
| epochs | 1.0 |
| learning_rate | 0.0002 |
| train_batch_size | 1 |
| eval_batch_size | 1 |
| gradient_accumulation_steps | 8 |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| use_4bit | True |
| step 25 training_loss | 0.246 |
| step 25 validation_loss | 0.331 |

### G2. Gemma LoRA Medium

| Metric | Value |
| --- | --- |
| base_model | google/gemma-2-2b-it |
| output_dir | models/adapters/gemma_2_2b_it_lora_combined_v1 |
| train_records | 4,000 |
| val_records | 400 |
| max_length | 1,536 |
| max_train_samples | 4,000 |
| max_val_samples | 400 |
| epochs | 1.0 |
| learning_rate | 0.0002 |
| train_batch_size | 1 |
| eval_batch_size | 1 |
| gradient_accumulation_steps | 8 |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| use_4bit | True |
| trainable_params | 20,766,720 |
| all_params | 2,635,108,608 |
| trainable_percent | 0.7881 |
| total_steps | 500 |
| runtime | 44m 51s |
| step 200 training_loss | 0.193 |
| step 200 validation_loss | 0.300 |
| step 400 training_loss | 0.132 |
| step 400 validation_loss | 0.264 |

### G3. Qwen3-32B QLoRA Training Run

| Metric | Value |
| --- | --- |
| base_model | Qwen/Qwen3-32B |
| output_dir | models/adapters/qwen3_32b_qlora_combined_v1 |
| train_jsonl | data/processed/finetune_train_combined.jsonl |
| val_jsonl | data/processed/finetune_val_combined.jsonl |
| train_records | 16,880 |
| val_records | 1,874 |
| max_length | 2,048 |
| max_train_samples | None |
| max_val_samples | None |
| epochs | 1.0 |
| learning_rate | 0.0001 |
| train_batch_size | 1 |
| eval_batch_size | 1 |
| gradient_accumulation_steps | 16 |
| lora_r | 32 |
| lora_alpha | 64 |
| lora_dropout | 0.05 |
| use_4bit | True |
| trainable_params | 268,435,456 |
| all_params | 33,030,558,720 |
| trainable_percent | 0.8127 |
| total_steps | 1,055 |
| runtime | 9h 18m 03s |
| step 250 training_loss | 0.086 |
| step 250 validation_loss | 0.164 |
| step 500 training_loss | 0.059 |
| step 500 validation_loss | 0.138 |
| step 750 training_loss | 0.048 |
| step 750 validation_loss | 0.126 |
| step 1000 training_loss | 0.033 |
| step 1000 validation_loss | 0.121 |
| step 1055 training_loss | 0.056 |
| step 1055 validation_loss | 0.121 |
| status | completed |

### G4. Qwen3-Embedding-8B LoRA Tuning

| Metric | Value |
| --- | --- |
| base_model | Qwen/Qwen3-Embedding-8B |
| output_dir | models/embedding_tuned/qwen3_embedding_8b_legal_lora_v1 |
| train_jsonl | data/processed/embedding_tune_train.jsonl |
| val_jsonl | data/processed/embedding_tune_val.jsonl |
| train_records | 1,854 |
| val_records | 205 |
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
| status | completed, not selected |

### G5. Qwen3-Reranker-8B LoRA v5 Tuning

| Metric | Value |
| --- | --- |
| base_model | Qwen/Qwen3-Reranker-8B |
| output_dir | models/reranker_tuned/qwen3_reranker_8b_legal_lora_v5 |
| train_jsonl | data/processed/reranker_tune_train.jsonl |
| val_jsonl | data/processed/reranker_tune_val.jsonl |
| train_records | 6,077 |
| val_records | 675 |
| batch_size | 1 |
| epochs | 1 |
| max_length | 2,048 |
| learning_rate | 1e-5 |
| lora_r | 32 |
| lora_alpha | 64 |
| lora_dropout | 0.05 |
| implementation | AutoModelForSequenceClassification + BCEWithLogitsLoss + TaskType.SEQ_CLS |
| status | completed, not selected |

## H. External Dataset Audits

### H1. Instructor External Dataset

| File | Count |
| --- | ---: |
| data/external/corpus.jsonl | 7,579 |
| data/external/llm.jsonl | 13,758 |
| data/external/embedding.jsonl | 2,059 |
| data/external/reranker.jsonl | 6,752 |
| data/external/gold_benchmark.json | 240 |
| data/external/rag_eval.json | 1,000 |

Usage:

- `llm.jsonl` was used for combined SFT data.
- `embedding.jsonl` and `reranker.jsonl` are available but currently deprioritized.
- External data is not used as official retrieval corpus.

### H2. CtnkyaABC/turkish-legal-rag-corpus Quick Audit

| Metric | Value |
| --- | ---: |
| dataset_total_rows | 6,350 |
| sampled_rows | 100 |
| sample_unique_source | 1 |
| source_in_sample | Türkiye Cumhuriyeti Anayasası |
| QA_nonempty_in_sample | 0 |
| URL_present_in_sample | 100 |
| Score=10_in_sample | 100 |
| chunk_strategy | statute_full |

Assessment:

- Useful as supplementary/custom demo data.
- Not better than our normalized official corpus for the main project.
- Lacks `doc_key`, `article_key`, `citation_label`, quality flags, and benchmark linkage.

## I. Current Best / Selected Components

Current best retrieval:

```text
Qwen/Qwen3-Embedding-8B dense top-30
Qwen/Qwen3-Reranker-8B top-10
```

Embedding and reranker tuning were both implemented and evaluated, but the tuned variants did not beat the zero-shot Qwen3 retrieval stack on the locked official-law benchmark. Therefore, the selected final retrieval remains the measured best-performing zero-shot combination.

Current LLM plan:

```text
Base RAG: Qwen/Qwen3-32B
Fine-tuned RAG: Qwen/Qwen3-32B + QLoRA
```

Generation uses precomputed reranker output:

```text
outputs/retrieval_eval/qwen3_embedding_8b_dense_top30_qwen3_reranker_8b_predictions_v1.csv
```

Reason:

- Loading Qwen3-32B and Qwen3-Reranker-8B at the same time caused CUDA OOM on a ~40GB GPU.
- Precomputing reranked context preserves the same retrieval/reranking result without loading the reranker during generation.

## J. Remaining Metrics To Add

When available, add:

1. Custom-data ingestion demo metrics.
2. Any final UI/demo latency measurements, if collected.

## K. Base vs Fine-Tuned RAG Semantic Similarity

Semantic similarity was computed with `Qwen/Qwen3-Embedding-8B` using cosine similarity between each `generated_answer` and `gold_answer`.

| System | answer_semantic_similarity | median | std |
| --- | ---: | ---: | ---: |
| Base RAG | 0.7890 | 0.8064 | 0.0835 |
| Fine-tuned RAG | 0.7908 | 0.8072 | 0.0911 |

Combined with the existing lexical metrics:

| Metric | Base RAG | Fine-tuned RAG | Change |
| --- | ---: | ---: | ---: |
| token_f1 | 0.1449 | 0.1914 | +0.0465 |
| rouge_l | 0.1233 | 0.1674 | +0.0441 |
| answer_semantic_similarity | 0.7890 | 0.7908 | +0.0018 |

Interpretation:

- Fine-tuning substantially improved lexical answer overlap and slightly improved semantic answer similarity.
- The semantic gain is small, but it supports the conclusion that the fine-tuned LLM moved answers closer to gold answers at the meaning level.
- User-safety interpretation: better semantic alignment can make answers feel more relevant and confidence-inducing, but legal systems must avoid unsupported confidence. Because citation-grounding decreased in the fine-tuned run, semantic similarity is reported together with grounding metrics rather than used as the only selection criterion.

## L. Leakage-Safe External-Mapped Fine-Tuning Metrics

### L1. External-Mapped Retrieval Ablation

Training source:

- `data/processed/external_mapped_embedding_train.jsonl`
- `data/processed/external_mapped_embedding_val.jsonl`
- `data/processed/external_mapped_reranker_train.jsonl`
- `data/processed/external_mapped_reranker_val.jsonl`

Adapters:

- `models/embedding_tuned/qwen3_embedding_8b_external_mapped_lora_v1`
- `models/reranker_tuned/qwen3_reranker_8b_external_mapped_lora_v1`

Index:

- `indexes/official_law_v3_qwen3_embedding_8b_external_mapped_lora_v1`

Index build:

| Metric | Value |
| --- | ---: |
| record_count | 24,473 |
| embedding_count | 24,473 |
| embedding_dim | 4,096 |

Retrieval metrics:

| System | article_hit@5 | article_hit@10 | article_recall@5 | article_recall@10 | article_mrr | article_ndcg@5 | article_ndcg@10 | doc_hit@10 | doc_mrr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| external_mapped_tuned_embedding_dense_only | 0.7421 | 0.7789 | 0.7421 | 0.7789 | 0.5859 | 0.6211 | 0.6336 | 0.9368 | 0.7578 |
| external_mapped_tuned_embedding_base_reranker | 0.1368 | 0.2737 | 0.1368 | 0.2737 | 0.0838 | 0.0830 | 0.1274 | 0.8316 | 0.5024 |
| external_mapped_tuned_embedding_tuned_reranker | 0.3474 | 0.5053 | 0.3474 | 0.5053 | 0.2102 | 0.2272 | 0.2793 | 0.8579 | 0.4609 |

Current interpretation:

- The external-mapped embedding LoRA adapter, reranker LoRA adapter, and tuned embedding index were built successfully.
- Dense-only tuned embedding is lower than the zero-shot Qwen3-Embedding-8B dense baseline on the locked official-law benchmark.
- The tuned embedding + base reranker combination produced a large ranking collapse.
- The tuned reranker partially recovered the ranking drop, but the fully tuned external-mapped retrieval stack did not beat the zero-shot Qwen3 retrieval stack.
