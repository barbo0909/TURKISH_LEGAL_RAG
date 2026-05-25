# Turkish Legal RAG - Instructor Quick Guide

This repository includes a custom evaluation path for instructor-provided documents and benchmarks.

## 0. Setup

Open a terminal in the project folder and run:

```bash
cd <project-folder>
pip install -r requirements.txt
```

## 1. Put your files in these folders

- Custom source documents:
  - `data/custom_docs/`
- Custom benchmark CSV:
  - `data/custom_benchmark/custom_benchmark.csv`

Supported custom document formats:

- `.txt`
- `.csv`
- `.jsonl`

## 2. Benchmark format

Minimum required column:

- `question`

Recommended columns:

- `question`
- `gold_answer`
- `gold_doc_keys`
- `gold_article_keys`

## 3. Run the evaluation

From the repository root, run:

```bash
python run_teacher_eval.py
```

This uses the strongest final system by default:

- `Qwen/Qwen3-Embedding-8B`
- `Qwen/Qwen3-Reranker-8B`
- `Qwen/Qwen3-32B`

Optional lighter modes:

```bash
python run_teacher_eval.py --no-generate-answers
python run_teacher_eval.py --no-reranker --no-generate-answers
```

## 4. What happens automatically

The system will automatically:

1. read the custom documents
2. convert them into the project retrieval format
3. build a custom index
4. read and normalize the benchmark
5. compute retrieval metrics if gold document/article fields are present
6. compute QA / citation / grounding metrics if gold answers are present

The instructor does not need to manually build the corpus, index, or metrics.

## 5. What you will see

The script prints the main metrics directly in the terminal.

It also writes output files here:

- `outputs/teacher_eval/teacher_eval_report.json`
- `outputs/teacher_eval/custom_retrieval_summary.json`
- `outputs/teacher_eval/custom_generation_summary.json` (if answer generation is enabled)

## 6. Notes

- If the machine is not strong enough for the full Qwen setup, use a GPU-backed environment.
- The notebook alternative is:
  - `notebooks/25_teacher_custom_data_single_notebook_demo.ipynb`

## 7. Google Drive / Colab usage

If the instructor prefers to run the project with Google Drive and Colab:

1. Upload the full repository folder to Google Drive.
2. Place custom documents in:
   - `data/custom_docs/`
3. Place the benchmark file at:
   - `data/custom_benchmark/custom_benchmark.csv`
4. Open the project in Colab.
5. Run one of the following:
   - `notebooks/25_teacher_custom_data_single_notebook_demo.ipynb`
   - or a Colab cell with:

```python
!python run_teacher_eval.py
```

If the benchmark includes:

- `question`
- `gold_answer`
- `gold_doc_keys`
- `gold_article_keys`

then the full retrieval, QA, citation-grounding, and hallucination-proxy metrics will be computed.
