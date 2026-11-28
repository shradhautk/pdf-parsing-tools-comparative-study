# A Comparative Study of PDF Parsing Tools Across Diverse Document Categories

Code for the paper **A Comparative Study of PDF Parsing Tools Across Diverse Document Categories** (N. S. Adhikari, S. Agarwal, arXiv 2024): [arXiv:2410.09871](https://arxiv.org/abs/2410.09871)

The study compares PDF parsing tools for text extraction across document categories, as a building block for retrieval-augmented generation (RAG) pipelines. Each tool is scored at the token level against annotated sample documents.

## Repository layout

| Folder | Contents |
| --- | --- |
| `pdfminer/`, `pdfplumber/`, `pdftext/`, `pymupdf/`, `pypdf2/`, `tika/`, `unstructured/` | Extraction script (`general_extraction.py`) and token-level scores for each tool |
| `utils/` | Preprocessing and evaluation code |
| `sample/` | Sample documents and annotations |

This repository contains the seven text-extraction tools listed above. The remaining tools compared in the paper are described in the paper itself.

## Setup

```bash
pip install -r requirements.txt
python <tool>/general_extraction.py
```

## Citation

```bibtex
@article{adhikari2024comparative,
  title={A Comparative Study of PDF Parsing Tools Across Diverse Document Categories},
  author={Adhikari, N. S. and Agarwal, Shradha},
  journal={arXiv preprint arXiv:2410.09871},
  year={2024}
}
```
