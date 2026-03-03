from ir_datasets import load
from pytest import raises

from lsr_benchmark import register_to_ir_datasets


def test_load_fails_for_non_existing_dataset() -> None:
    with raises(ValueError):
        register_to_ir_datasets("lsr-benchmark")


def test_load_webis_splade_on_dl_19() -> None:
    register_to_ir_datasets("msmarco-passage/trec-dl-2019/judged")
    irds = load("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")

    doc_embeddings = irds.doc_embeddings(model_name="lightning-ir/webis/splade")
    query_embeddings = irds.query_embeddings(model_name="lightning-ir/webis/splade")

    assert doc_embeddings is not None
    assert query_embeddings is not None


def test_load_on_robust04() -> None:
    register_to_ir_datasets("disks45/nocr/trec-robust-2004/fold1")
    irds = load("lsr-benchmark/disks45/nocr/trec-robust-2004/fold1")

    doc_embeddings = irds.doc_embeddings(model_name="lightning-ir/webis/splade")
    query_embeddings = irds.query_embeddings(model_name="lightning-ir/webis/splade")

    assert doc_embeddings is not None
    assert query_embeddings is not None
