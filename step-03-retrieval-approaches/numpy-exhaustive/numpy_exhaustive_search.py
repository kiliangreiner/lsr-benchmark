#!/usr/bin/env python3
import gzip
from pathlib import Path
import click
import numpy as np
from tirex_tracker import ExportFormat, register_metadata, tracking
import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


def retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k):
    """Exhaustive Cosine Similarity Retrieval with numpy."""
    results = []
    for query_id, query_vec in zip(query_ids, query_embeddings):
        scores = np.dot(doc_embeddings, query_vec) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
        )
        topk_indices = np.argsort(scores)[::-1][:k]
        ranking = []
        for idx in topk_indices:
            if scores[idx] > 0:
                ranking.append((query_id, float(scores[idx]), doc_ids[idx]))
        results.append(ranking)
    return results


def to_numpy_array(embeddings):
    dim = None
    embedding_ids, np_embeddings = [], []
    for embedding_id, tokens, values in embeddings:
        embedding_ids.append(embedding_id)
        if dim is None:
            dim = max([int(t) for t in tokens]) + 1
        vec = np.zeros(dim)
        for t, v in zip(tokens, values):
            vec[int(t)] = v
        np_embeddings.append(vec)
    return embedding_ids, np.array(np_embeddings)


@retrieve_command()
def main(dataset, embedding, output, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata({
        "actor": {"team": "reneuir-baselines"},
        "tag": f"numpy-exhaustive-{embedding.replace('/', '-')}-{k}",
    })

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        doc_ids, doc_embeddings = to_numpy_array(load_embeddings(dataset, embedding, "doc"))

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        query_ids, query_embeddings = to_numpy_array(load_embeddings(dataset, embedding, "query"))
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k)

    with gzip.open(output / "run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} numpy-exhaustive\n")
                rank += 1


if __name__ == "__main__":
    main()
