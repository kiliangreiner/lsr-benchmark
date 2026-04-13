#!/usr/bin/env python3
from pathlib import Path
import gzip
import json

import click
import numpy as np
from sentence_transformers import SentenceTransformer
from tirex_tracker import register_metadata

import lsr_benchmark
from lsr_benchmark.click import option_lsr_dataset


def convert_embeddings_dense(embeddings: np.ndarray):
    n_docs, n_dims = embeddings.shape
    row_idcs = np.repeat(np.arange(n_docs), n_dims)
    col_idcs = np.tile(np.arange(n_dims), n_docs)
    values = embeddings.flatten()
    row_indices = np.bincount(row_idcs + 1, minlength=n_docs + 1).cumsum()
    return values, col_idcs, row_indices

def embedd_text_with_model(model, texts, ids, output):
    with tracking(export_file_path= str(output).replace("-embeddings.npz", "-ir-metadata.yml")):
        embeddings = module.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        data, indices, indptr = convert_embeddings_dense(embeddings)
        np.savez_compressed(
            text_type_save_dir / f"{text_type}-embeddings.npz",
            data=data,
            indices=indices,
            indptr=indptr,
        )
        Path(str(output).replace("-embeddings.npz", "-ids.txt")).write_text("\n".join(ids))


@click.command()
@option_lsr_dataset()
@click.option("--model", type=str, required=True, help="The Sentence Transformers model.")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
def main(dataset: str, model: str, batch_size: int, output: Path):
    lsr_benchmark.register_to_ir_datasets(dataset)

    module = SentenceTransformer(model)

    register_metadata({"actor": {"team": "sentence-transformers"}, "tag": model.replace('/', '-')})

    ir_dataset = ir_dataset.load(dataset)

    for text_type in ["query", "doc"]:
        text_type_save_dir = Path(output) / text_type
        text_type_save_dir.mkdir(parents=True, exist_ok=True)

        if text_type == "query":
            records = [json.loads(l) for l in (dataset_path / "queries.jsonl").open()]
            texts = [r["query"] for r in records]
            ids = [r["qid"] for r in records]
        else:
            records = [json.loads(l) for l in gzip.open(dataset_path / "corpus.jsonl.gz", "rt")]
            texts = [r["default_text"] for r in records]
            ids = [r["doc_id"] for r in records]

        embedd_text_with_model(module, texts, ids, text_type_save_dir / f"{text_type}-embeddings.npz")

if __name__ == "__main__":
    main()
