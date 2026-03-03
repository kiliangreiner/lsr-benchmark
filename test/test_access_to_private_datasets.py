from ir_datasets import load, Dataset
from pytest import raises

from lsr_benchmark import register_to_ir_datasets


def load_ds(ds_id) -> Dataset:
    register_to_ir_datasets(ds_id)
    return load(f"lsr-benchmark/{ds_id}")


def test_private_dataset_can_be_loaded() -> None:
    ds = load_ds("clueweb09/en/trec-web-2009")
    assert ds is not None


def test_private_dataset_fails_to_access_documents() -> None:
    ds = load_ds("clueweb09/en/trec-web-2009")
    with raises(ValueError, match="The dataset trec-18-web-20251008-test is private, you can not access the raw data."):
        next(ds.docs_iter())


def test_public_dataset_can_access_documents() -> None:
    ds = load_ds("msmarco-passage/trec-dl-2019/judged")
    assert 32123 == sum(1 for _ in ds.docs_iter())


def test_public_dataset_can_access_qrels() -> None:
    ds = load_ds("msmarco-passage/trec-dl-2019/judged")
    assert 9260 == sum(1 for _ in ds.qrels_iter())


def test_qrels_can_be_accessed_on_private_dataset() -> None:
    ds = load_ds("clueweb09/en/trec-web-2009")
    assert 23601 == sum(1 for _ in ds.qrels_iter())
