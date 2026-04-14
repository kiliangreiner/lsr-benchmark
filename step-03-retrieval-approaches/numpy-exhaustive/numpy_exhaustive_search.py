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
    """Exhaustive Cosine Similarity Retrieval mit numpy."""
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


@retrieve_command()
def main(dataset, embedding, output, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata({
        "actor": {"team": "reneuir-baselines"},
        "tag": f"numpy-exhaustive-{embedding.replace('/', '-')}-{k}",
    })

    doc_ids, doc_embeddings = [], []
    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for doc_id, tokens, values in load_embeddings("dummy", embedding, "doc"):
            doc_ids.append(doc_id)
            vec = np.zeros(1024)
            for t, v in zip(tokens, values):
                vec[int(t)] = v
            doc_embeddings.append(vec)
    doc_embeddings = np.array(doc_embeddings)

    query_ids, query_embeddings = [], []
    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for query_id, tokens, values in load_embeddings("dummy", embedding, "query"):
            query_ids.append(query_id)
            vec = np.zeros(1024)
            for t, v in zip(tokens, values):
                vec[int(t)] = v
            query_embeddings.append(vec)
        query_embeddings = np.array(query_embeddings)
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k)

    with gzip.open(output / "run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} numpy-exhaustive\n")
                rank += 1


if __name__ == "__main__":
    main()
