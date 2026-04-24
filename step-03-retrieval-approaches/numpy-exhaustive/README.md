# Numpy Exhaustive Retrieval

Exhaustive nearest-neighbor retrieval using cosine similarity implemented with numpy only, without additional dependencies.

## Usage

```bash
python numpy_exhaustive_search.py \
    --dataset <path-to-dataset> \
    --embedding <path-to-embeddings> \
    --output <output-dir> \
    --k 1000
```

## Run Unit Tests

```bash
pip install pytest numpy
python -m pytest test_retrieval.py -v
```
