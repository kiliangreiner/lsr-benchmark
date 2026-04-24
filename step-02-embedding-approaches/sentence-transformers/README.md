# Sentence Transformers Embedding Approach

This approach uses [Sentence Transformers](https://www.sbert.net/) to embed queries and documents.

## Usage

```bash
python sentence_transformers_embeddings.py \
    --dataset <path-to-dataset> \
    --model <model-name> \
    --batch_size 32 \
    --output <output-dir>
```

## Run Unit Tests

```bash
pip install pytest sentence-transformers
python -m pytest test_embeddings.py -v
```
