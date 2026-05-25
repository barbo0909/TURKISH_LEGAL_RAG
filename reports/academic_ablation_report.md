# Turkish Legal RAG for Gold Q+A+Doc Evaluation: An Academic-Style Ablation Report

## Abstract

This report presents an end-to-end Turkish legal Retrieval-Augmented Generation (RAG) study designed around a gold benchmark containing questions, verified answers, and gold legal sources at the document and article level. The project goal was not only to improve answer quality, but also to improve legal grounding, citation reliability, and hallucination resistance. For that reason, the study evaluates corpus normalization, retrieval design, reranking, embedding tuning, reranker tuning, and LLM fine-tuning within a unified ablation framework.

The strongest final retrieval stack was `Qwen/Qwen3-Embedding-8B dense top-30 + Qwen/Qwen3-Reranker-8B top-10`, while the safest final generation system was `Qwen/Qwen3-32B Base` on top of that retrieval stack. QLoRA fine-tuning improved lexical overlap metrics, but the base Qwen system remained stronger on citation-grounding and hallucination-proxy metrics.

## 1. Introduction

Legal-domain RAG systems should not be evaluated solely by whether they produce plausible answers. In a legal setting, the system must retrieve the correct law and ideally the correct article, then produce an answer that remains faithful to those retrieved sources. This project therefore focuses on a stricter setup than a standard QA benchmark: a gold `Question + Answer + Document/Article` benchmark, article-level retrieval metrics, citation-aware prompting, and hallucination-oriented analysis.

The work was organized around the instructor requirements:

1. baseline RAG
2. embedding tuning
3. reranker
4. LLM fine-tuning
5. fully optimized system

The final deliverable is both a set of quantitative results and a reproducible pipeline that can operate on instructor-provided custom legal collections.

## 2. Dataset and Corpus Construction

### 2.1 Corpus Normalization

The project starts from a curated raw corpus:

- input: `data/raw/legal_documents_curated.csv`

The normalization pipeline extracts official-law article records into the main retrieval corpus and separates QA-like rows into an auxiliary file.

**Table 1. Corpus normalization summary**

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

The rejected/review pool was also audited so that the main retrieval corpus would remain article-centric and legally grounded.

**Table 2. Rejected / review quality flags**

| quality_flag | Count |
| --- | ---: |
| duplicate_conflict | 6,474 |
| amendment_fragment | 2,334 |
| duplicate_candidate | 939 |
| table_fragment | 3 |
| metadata_only | 2 |
| empty_or_too_short | 1 |

**Table 3. Main corpus quality flags**

| quality_flag | Count |
| --- | ---: |
| valid_article | 24,136 |
| short_but_valid | 337 |

This normalization step is foundational: later retrieval ablations show that one of the largest early gains came from moving away from a mixed Kaggle-style corpus and toward a normalized official-law article corpus.

### 2.2 Gold Benchmark

The project uses a locked benchmark:

- benchmark file: `data/benchmark/gold_benchmark_v1.csv`

It includes gold answer text and gold document/article supervision.

**Table 4. Benchmark summary**

| Metric | Value |
| --- | ---: |
| benchmark_questions | 190 |
| benchmark_columns | 16 |
| missing_article_keys | 0 |
| gold_article_coverage_in_corpus | 190 / 190 |

This satisfies the `Gold Q + A + Doc` requirement in the evaluation rubric.

## 3. Evaluation Design

The project uses a layered evaluation design.

### 3.1 Retrieval Metrics

Retrieval was measured with:

- `doc_hit@5`, `doc_hit@10`
- `article_hit@5`, `article_hit@10`
- `doc_recall@5`, `doc_recall@10`
- `article_recall@5`, `article_recall@10`
- `doc_mrr`, `article_mrr`
- `doc_ndcg@5`, `doc_ndcg@10`
- `article_ndcg@5`, `article_ndcg@10`

### 3.2 Generation and Grounding Metrics

Generation and answer-grounding were measured with:

- `exact_match`
- `token_f1`
- `rouge_l`
- `retrieval_gold_available`
- `citation_present`
- `citation_gold_match`
- `grounded_citation_score`
- `unsupported_or_missing_citation`

### 3.3 Hallucination Analysis

Hallucination analysis was handled through deterministic proxies and explicit error categories:

- `retrieval_miss`
- `missing_citation`
- `wrong_or_unsupported_citation`
- `low_answer_overlap`
- `acceptable_automatic`

### 3.4 LLM Judge

To complement deterministic metrics, a full LLM-judge evaluation was run on the best base Qwen system with the following dimensions:

- correctness
- faithfulness
- citation support
- hallucination risk

## 4. Compute Environment and Hyperparameters

### 4.1 GPU Usage

The heavy experiments in this project were run in a Google Colab GPU environment. The exact accelerator type was not logged consistently across all sessions, so the report avoids over-claiming a specific hardware SKU when it cannot be verified from logs. Local Windows runs were used mainly for repository organization, report preparation, and notebook/UI cleanup rather than for the main heavy training/evaluation runs.

**Table 5. GPU usage by experiment family**

| Experiment | GPU / environment | Note |
| --- | --- | --- |
| Corpus indexing / dense embedding build | Google Colab GPU environment | CUDA-enabled Colab session; notebook logic uses `cuda` when available and `batch_size=32` on GPU |
| Dense retrieval evaluation | Google Colab GPU environment | Retrieval notebooks and scripts select `cuda` if available |
| BM25 retrieval evaluation | CPU or Google Colab runtime | Sparse retrieval is not GPU-dependent |
| Reranker evaluation | Google Colab GPU environment | Qwen3-Reranker-8B and MiniLM reranker evaluations were executed in CUDA-enabled notebook sessions |
| Gemma base / LoRA generation evaluation | Google Colab GPU environment | 4-bit loading used in generation evaluation notebooks when CUDA was available |
| Qwen3-32B base generation | Google Colab GPU environment | Heavy generation runs used 4-bit loading and precomputed reranked retrieval outputs |
| Qwen3-32B QLoRA fine-tuning | Google Colab GPU environment | QLoRA adapter training for the final same-LLM comparison |
| Embedding LoRA fine-tuning | Google Colab GPU environment | Qwen3-Embedding-8B legal adaptation run |
| Reranker LoRA fine-tuning | Google Colab GPU environment | Qwen3-Reranker-8B legal adaptation run |
| LLM judge evaluation | Google Colab GPU environment | Qwen3-32B judge loaded in 4-bit mode when CUDA was available |

### 4.2 Main Hyperparameters and Configuration

The project uses a mixture of notebook-level defaults, saved JSON run configurations, and model-specific fine-tuning configs. Where a parameter was not explicitly logged in a run artifact, the report marks it as `default` or points to notebook/script configuration rather than inventing a value.

**Table 6. Main system hyperparameters**

| Component | Model | Hyperparameters |
| --- | --- | --- |
| Dense retriever (final) | `Qwen/Qwen3-Embedding-8B` | `candidate_k=30`, final reranked `top_k=10`; corpus indexing `batch_size=32` on GPU (`8` on CPU) |
| Dense retriever (BGE baseline) | `BAAI/bge-m3` | retrieval evaluation with `top_k=10`, `candidate_k=30` |
| BM25 | BM25 lexical index | retrieval evaluation with `top_k=10`, `candidate_k=30` |
| Hybrid retrieval | BGE-M3 + BM25 | weighted fusion sweep; main reported hybrid used dense-heavy weighting (see retrieval notebooks / summaries) |
| Reranker (final) | `Qwen/Qwen3-Reranker-8B` | rerank dense candidates with `candidate_k=30`, `top_k=10`; eval notebook uses `batch_size=16` on GPU (`4` on CPU) |
| Reranker (UI/demo) | `Qwen/Qwen3-Reranker-8B` | demo defaults: `candidate_k=30`, `top_k_context=10`, `reranker_batch_size=4` |
| Base LLM generation (Gemma) | `google/gemma-2-2b-it` | `top_k_context=10`, `candidate_k=30`, `max_new_tokens=384`, `temperature=0.2`, `top_p=0.9`, `max_context_chars=9000`, `load_in_4bit=True` |
| Final base LLM generation | `Qwen/Qwen3-32B` | final UI/demo defaults: `candidate_k=30`, `top_k_context=10`, `max_context_chars=9000`, `max_new_tokens=384`; generation uses citation-aware prompting and 4-bit loading in CUDA sessions |
| LLM judge | `Qwen/Qwen3-32B` | `load_in_4bit=True`, `max_new_tokens=256` in the judge pipeline |
| QLoRA generation comparison | `Qwen/Qwen3-32B` + adapter | same retrieval stack as final base system; adapter path `models/adapters/qwen3_32b_qlora_combined_v1` |

**Table 7. Fine-tuning hyperparameters**

| Fine-tuning component | Model | Hyperparameters |
| --- | --- | --- |
| Gemma LoRA smoke | `google/gemma-2-2b-it` | `epochs=1.0`, `learning_rate=2e-4`, `train_batch_size=1`, `eval_batch_size=1`, `gradient_accumulation_steps=8`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05` |
| Gemma LoRA combined | `google/gemma-2-2b-it` | `epochs=1.0`, `learning_rate=2e-4`, `train_batch_size=1`, `eval_batch_size=1`, `gradient_accumulation_steps=8`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05` |
| Qwen3-32B QLoRA | `Qwen/Qwen3-32B` | `epochs=1.0`, `learning_rate=1e-4`, `train_batch_size=1`, `eval_batch_size=1`, `gradient_accumulation_steps=16`, `max_length=2048`, `lora_r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `use_4bit=True` |
| Embedding LoRA tuning | `Qwen/Qwen3-Embedding-8B` | `epochs=1`, `batch_size=1`, `warmup_ratio=0.1`, `warmup_steps=185`, `max_seq_length=1024`, `use_hard_negative=True`, `lora_r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `torch_dtype=bfloat16` |
| Reranker LoRA tuning (v5) | `Qwen/Qwen3-Reranker-8B` | `epochs=1`, `batch_size=1`, `learning_rate=1e-5`, `warmup_ratio=0.1`, `warmup_steps=607`, `gradient_accumulation_steps=1`, `max_length=2048`, `lora_r=32`, `lora_alpha=64`, `lora_dropout=0.05` |

## 5. Experimental Timeline

**Table 8. End-to-end experiment timeline**

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

## 6. Retrieval Ablation Results

### 6.1 Old Kaggle-only Baseline

The oldest baseline is important because it reveals the initial problem: document/source retrieval can look acceptable while article-level retrieval remains almost unusable.

**Table 9. Old Kaggle-only baseline**

| Metric | Value |
| --- | ---: |
| source_hit@5 | ~0.795 |
| article_hit@5 | ~0.053 |
| joint_source_article_hit@5 | ~0.047 |
| mrr_source_article | ~0.035 |
| ndcg_source_article@5 | ~0.038 |

The article-level numbers are extremely weak. This justified the full corpus redesign around official article records.

### 6.2 Main Retrieval Experiments

**Table 7. Retrieval ablation results**

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

### 5.3 Interpretation of Retrieval Results

The retrieval chain tells a clear story.

First, corpus normalization produced the largest early jump: article-level retrieval moved from nearly unusable Kaggle-only behavior to a solid official-law baseline. Second, stronger dense retrieval with `Qwen3-Embedding-8B` improved the article-level metrics again. Third, the best overall article-level retrieval was reached by adding `Qwen3-Reranker-8B` on top of Qwen dense retrieval.

The tuning experiments are also informative. Neither tuned embedding nor tuned reranker surpassed the strongest zero-shot Qwen retrieval stack. This is a meaningful result rather than a failure: it shows that domain adaptation must be validated empirically rather than assumed to help by default.

### 5.4 Retrieval Improvement Chain

**Table 8. Retrieval improvement chain**

| Step | article_hit@5 | Relative note |
| --- | ---: | --- |
| Old Kaggle-only | ~0.053 | very weak article retrieval |
| Official corpus + BGE-M3 dense | 0.674 | corpus normalization gave the largest early jump |
| Qwen3-Embedding-8B dense | 0.774 | stronger multilingual embedding |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 0.837 | final selected retrieval stack |
| Qwen3-Embedding-8B LoRA tuned dense | 0.742 | embedding tuning did not improve official article retrieval |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B LoRA v5 | 0.737 | reranker tuning did not improve over zero-shot Qwen3 reranker |
| Qwen3-Embedding-8B LoRA tuned + Qwen3-Reranker-8B LoRA v5 | 0.705 | fully tuned retrieval stack did not beat the zero-shot final stack |

Improvement from old Kaggle-only to the final retrieval stack:

```text
article_hit@5: ~0.053 -> 0.837
approximate improvement: 15.8x
```

## 7. Generation and Grounding Results

### 6.1 Main Generation Experiments

**Table 9. Main generation and grounding results**

| Experiment | Questions | Retriever | Reranker | LLM | Fine-tune | exact_match | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E6 Gemma Base RAG | 190 | BGE-M3 dense | none | Gemma-2-2B-it | none | 0.000 | 0.144 | 0.128 | 0.732 | 0.568 | 0.337 | 0.289 | 0.574 |
| E7 Gemma LoRA RAG | 190 | BGE-M3 dense | none | Gemma-2-2B-it | LoRA | 0.000 | 0.127 | 0.109 | 0.732 | 0.642 | 0.179 | 0.163 | 0.563 |
| E10 Qwen3-32B Base RAG smoke | 10 | Qwen3-Embedding-8B | Qwen3-Reranker-8B, precomputed | Qwen3-32B | none | 0.000 | 0.164 | 0.133 | 0.900 | 1.000 | 0.900 | 0.900 | 0.100 |
| E10 Qwen3-32B Base RAG full | 190 | Qwen3-Embedding-8B | Qwen3-Reranker-8B, precomputed | Qwen3-32B | none | 0.000 | 0.145 | 0.123 | 0.874 | 0.984 | 0.821 | 0.805 | 0.142 |
| E11 Qwen3-32B QLoRA RAG | 190 | Qwen3-Embedding-8B | Qwen3-Reranker-8B, precomputed | Qwen3-32B | QLoRA | 0.000 | 0.191 | 0.167 | 0.874 | 0.932 | 0.742 | 0.721 | 0.195 |

### 6.2 Interpretation of Generation Results

The Gemma experiments show that fine-tuning did not automatically produce better legal QA. In fact, the Gemma LoRA run reduced both answer-overlap and grounding quality.

The Qwen experiments are more subtle. QLoRA increased `token_f1` from `0.145` to `0.191` and `rouge_l` from `0.123` to `0.167`, which indicates better lexical overlap with gold answers. However, this came with lower citation quality and weaker grounding:

- `citation_gold_match`: `0.821 -> 0.742`
- `grounded_citation_score`: `0.805 -> 0.721`
- `unsupported_or_missing_citation`: `0.142 -> 0.195`

This is one of the central findings of the project: answer similarity improved after QLoRA, but legal grounding and citation reliability became weaker. Because the project prioritizes legally grounded QA rather than free-form paraphrase quality alone, the final selected generation model remains `Qwen3-32B Base`.

### 6.3 Topic-Level Qwen Comparison

The topic-level breakdown shows that the QLoRA model generally improves overlap metrics across topics, but not uniformly in citation-grounding terms. The strongest base-model citation behavior appears in topics such as `Is Hukuku` and `Medeni Hukuk`, while `Ticaret Hukuku` remains the weakest topic for both answer quality and grounding.

## 8. Hallucination and Error Analysis

The project uses deterministic hallucination/error proxies based on retrieval availability, citation presence, gold citation match, grounded citation score, and answer-overlap.

**Table 10. Final hallucination and error analysis**

| system | status | exact_match | token_f1 | rouge_l | retrieval_gold_available | citation_present | citation_gold_match | grounded_citation_score | unsupported_or_missing_citation | error_missing_citation | error_retrieval_miss | error_acceptable_automatic | error_low_answer_overlap | error_wrong_or_unsupported_citation |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma_base_rag | ok | 0 | 0.143630 | 0.127552 | 0.731579 | 0.568421 | 0.336842 | 0.289474 | 0.573684 | 58 | 51 | 28 | 27 | 26 |
| gemma_finetuned_rag | ok | 0 | 0.126803 | 0.109211 | 0.731579 | 0.642105 | 0.178947 | 0.163158 | 0.563158 | 56 | 51 | 15 | 16 | 52 |
| qwen3_32b_base_rag | ok | 0 | 0.144925 | 0.123274 | 0.873684 | 0.984211 | 0.821053 | 0.805263 | 0.142105 | 3 | 24 | 75 | 78 | 10 |
| qwen3_32b_finetuned_rag | ok | 0 | 0.191439 | 0.167383 | 0.873684 | 0.931579 | 0.742105 | 0.721053 | 0.194737 | 13 | 24 | 95 | 42 | 16 |

### 7.1 Interpretation of Hallucination Results

The deterministic error analysis strongly supports the selection of the base Qwen system. Compared with the fine-tuned Qwen system, the base model has:

- fewer missing citations (`3` vs `13`)
- fewer wrong or unsupported citations (`10` vs `16`)
- stronger citation grounding (`0.805` vs `0.721`)
- lower unsupported or missing citation rate (`0.142` vs `0.195`)

The fine-tuned model does reduce `low_answer_overlap`, but from a legal QA standpoint that alone is not enough to justify preferring it over the more grounded base model.

## 9. Full LLM Judge Evaluation

To complement deterministic evaluation, the best base Qwen system was judged by `Qwen/Qwen3-32B` over the full 190-question benchmark.

**Table 11. Full LLM judge evaluation**

| Metric | Value |
| --- | ---: |
| correctness | 4.053 / 5 |
| faithfulness | 3.789 / 5 |
| citation_support | 3.468 / 5 |
| hallucination_risk | 1.795 / 5 |

### 8.1 Interpretation of LLM Judge Results

These scores indicate that the final base system:

- produces generally correct answers
- stays relatively faithful to retrieved material
- has moderate-to-good citation support
- maintains low hallucination risk

The LLM judge therefore supports the deterministic grounding analysis rather than contradicting it.

## 10. Final Selected System

The final selected architecture is:

- corpus: normalized official-law article corpus v3
- benchmark: locked 190-question Turkish legal Q+A+Doc benchmark
- retriever: `Qwen/Qwen3-Embedding-8B` dense top-30
- reranker: `Qwen/Qwen3-Reranker-8B` top-10
- generator: `Qwen/Qwen3-32B Base`

### 9.1 Final Headline Metrics

**Table 12. Final selected system headline metrics**

| Metric group | Key result |
| --- | --- |
| Retrieval | article_hit@5=0.837, article_hit@10=0.874, article_mrr=0.701, article_ndcg@5=0.731 |
| QA / citation | token_f1=0.145, rouge_l=0.123, citation_gold_match=0.821, grounded_citation_score=0.805 |
| Hallucination | unsupported_or_missing_citation=0.142, LLM judge hallucination_risk=1.795/5 |
| LLM judge | correctness=4.053/5, faithfulness=3.789/5, citation_support=3.468/5 |

## 11. Mapping to the Instructor Rubric

The instructor rubric required:

1. a 150-300 question Turkish legal gold benchmark
2. baseline RAG
3. embedding tuning
4. reranker
5. LLM fine-tuning
6. fully optimized system
7. retrieval metrics
8. QA metrics
9. faithfulness / citation accuracy / hallucination analysis
10. custom-data support

This project satisfies those items as follows:

- benchmark: `190` questions with gold question, answer, and article/document fields
- baseline RAG: Gemma Base, Qwen3 Base
- embedding tuning: Qwen3-Embedding-8B LoRA tuned retrieval
- reranker: MiniLM ablation, Qwen3 reranker, Qwen3 reranker tuning
- LLM fine-tuning: Gemma LoRA and Qwen3 QLoRA
- fully optimized system: Qwen3 embedding + Qwen3 reranker + Qwen3 base generation
- retrieval metrics: Recall@5/10, MRR, nDCG
- QA metrics: EM, F1, ROUGE-L
- grounding metrics: citation presence, citation gold match, grounded citation score
- hallucination analysis: deterministic error analysis + full LLM judge
- custom data support: `25_teacher_custom_data_single_notebook_demo.ipynb`

## 12. Discussion

The most important technical lesson from this study is that system quality did not improve monotonically with every fine-tuning step. Instead, the strongest gains came from:

1. normalizing the corpus into official-law article records
2. replacing older retrieval models with stronger Qwen retrieval components
3. explicitly evaluating article-level retrieval rather than only source-level retrieval

This matters because a legal RAG system can look deceptively strong if only source-level metrics or answer-overlap metrics are considered. The Kaggle-only baseline already hinted at this problem: document/source retrieval looked decent, but article-level grounding was almost absent. Similarly, QLoRA improved overlap metrics but weakened citation-grounding. These findings justify why the project selected the strongest grounded system rather than the system with the highest lexical-overlap score alone.

## 13. Conclusion

This project demonstrates that a Turkish legal RAG system can be significantly improved through corpus normalization, strong dense retrieval, and reranking, even when subsequent fine-tuning steps do not always produce monotonic gains. The final system improves article-level retrieval from approximately `0.053` article_hit@5 in the old Kaggle-only baseline to `0.837` in the selected final setup, while maintaining strong citation-grounding and low hallucination risk.

The final selected system is therefore not simply the most heavily tuned system, but the most reliable one under legal grounding criteria:

`official-law v3 + Qwen3-Embedding-8B dense top-30 + Qwen3-Reranker-8B top-10 + Qwen3-32B Base`

That choice is fully supported by retrieval, generation, citation, hallucination, and LLM-judge evidence.
