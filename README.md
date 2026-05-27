# Turkish Legal RAG - Instructor Quick Guide

This repository includes a custom evaluation path for instructor-provided documents and benchmark files.

The recommended instructor workflow is the Colab notebook:

- `notebooks/25_teacher_custom_data_single_notebook_demo.ipynb`

This notebook reads instructor-provided documents, builds a custom RAG index, evaluates the benchmark, prints the main metrics directly in the notebook, and optionally opens a small demo UI.

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

### Option A: Google Drive / Colab recommended

1. Download or clone this GitHub repository.
2. Upload the full project folder to Google Drive.
3. Rename the Drive folder to:

```text
rag
```

The expected Colab path is:

```text
/content/drive/MyDrive/rag
```

4. Put instructor documents under:

```text
/content/drive/MyDrive/rag/data/custom_docs/
```

5. Put the instructor benchmark file here:

```text
/content/drive/MyDrive/rag/data/custom_benchmark/custom_benchmark.csv
```

6. Open and run:

```text
notebooks/25_teacher_custom_data_single_notebook_demo.ipynb
```

The notebook will print the retrieval and QA/citation/grounding metrics directly in the output cells. The final metric summary cell shows the main results in one compact table.

### Option B: Local / terminal

```bash
cd <project-folder>
pip install -r requirements.txt
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

The script/notebook prints the main metrics directly in the terminal or Colab output.

The terminal script writes output files here:

- `outputs/teacher_eval/teacher_eval_report.json`
- `outputs/teacher_eval/custom_retrieval_summary.json`
- `outputs/teacher_eval/custom_generation_summary.json` (if answer generation is enabled)

The Colab notebook writes notebook-specific metric files here:

- `outputs/retrieval_eval/custom_teacher_demo_retrieval_metrics.csv`
- `outputs/retrieval_eval/custom_teacher_demo_retrieval_summary.json`
- `outputs/generation_eval/custom_teacher_demo_qa_metrics.csv`
- `outputs/generation_eval/custom_teacher_demo_qa_summary.json`

## 6. Notes

- If the machine is not strong enough for the full Qwen setup, use a GPU-backed environment.
- The teacher-facing notebook is:
  - `notebooks/25_teacher_custom_data_single_notebook_demo.ipynb`
- The final official system UI notebook is:
  - `notebooks/24_final_system_rag_ui_demo.ipynb`

## 7. Google Drive / Colab usage

If running a simple Colab cell instead of the notebook, use:

Important: the Drive folder should be named `rag`. If a different folder name is used, update `DRIVE_ROOT` inside the notebook.

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/rag
!pip install -r requirements.txt
!python run_teacher_eval.py
```

If the benchmark includes:

- `question`
- `gold_answer`
- `gold_doc_keys`
- `gold_article_keys`

then the full retrieval, QA, citation-grounding, and hallucination-proxy metrics will be computed.
