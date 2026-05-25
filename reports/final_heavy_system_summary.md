# Final Heavy System Summary

## Retrieval Comparison

| experiment                             |   doc_hit@5 |   doc_hit@10 |   article_hit@5 |   article_hit@10 |   doc_mrr |   article_mrr |   article_ndcg@5 |   article_ndcg@10 |
|:---------------------------------------|------------:|-------------:|----------------:|-----------------:|----------:|--------------:|-----------------:|------------------:|
| BGE-M3 dense                           |    0.884211 |     0.9      |        0.673684 |         0.731579 |  0.764231 |      0.519925 |         0.552414 |          0.57166  |
| Qwen3-Embedding-8B dense               |    0.968421 |     0.978947 |        0.773684 |         0.836842 |  0.832897 |      0.624144 |         0.652632 |          0.67374  |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B |    0.968421 |     0.978947 |        0.836842 |         0.873684 |  0.869348 |      0.700921 |         0.731402 |          0.743439 |


## Generation Comparison

| experiment                           |   exact_match |   token_f1 |    rouge_l |   retrieval_gold_available |   citation_present |   citation_gold_match |   grounded_citation_score |   unsupported_or_missing_citation |
|:-------------------------------------|--------------:|-----------:|-----------:|---------------------------:|-------------------:|----------------------:|--------------------------:|----------------------------------:|
| Gemma-2-2B Base RAG                  |           nan | nan        | nan        |                 nan        |         nan        |            nan        |                nan        |                        nan        |
| Gemma-2-2B LoRA RAG                  |           nan | nan        | nan        |                 nan        |         nan        |            nan        |                nan        |                        nan        |
| Qwen3-32B Base RAG + best retrieval  |             0 |   0.144925 |   0.123274 |                   0.873684 |           0.984211 |              0.821053 |                  0.805263 |                          0.142105 |
| Qwen3-32B QLoRA RAG + best retrieval |             0 |   0.191439 |   0.167383 |                   0.873684 |           0.931579 |              0.742105 |                  0.721053 |                          0.194737 |


## Selected Final Architecture

- Corpus: normalized official-law article corpus v3

- Benchmark: locked 190-question Q-A-Doc Turkish legal benchmark

- Retriever: Qwen/Qwen3-Embedding-8B dense top-30

- Reranker: Qwen/Qwen3-Reranker-8B top-10 context

- Base LLM: Qwen/Qwen3-32B

- Fine-tuned LLM: Qwen/Qwen3-32B + QLoRA
