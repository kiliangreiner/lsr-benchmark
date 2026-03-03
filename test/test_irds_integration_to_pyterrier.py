from pathlib import Path
from lsr_benchmark import register_to_ir_datasets

from pyterrier import get_dataset
from pyterrier.datasets import Dataset


def load_dataset_with_pyterrier(irds_id) -> Dataset:
    return get_dataset(f"irds:{irds_id}")


def test_that_original_dataset_can_be_loaded() -> None:
    dataset = load_dataset_with_pyterrier("clueweb09/en/trec-web-2009")
    assert dataset is not None


def test_from_local_directory() -> None:
    resource_dir = str(Path(__file__).parent / "resources" / "example-dataset")
    register_to_ir_datasets(resource_dir)
    ds = load_dataset_with_pyterrier("lsr-benchmark/" + resource_dir)

    assert 3 == len(ds.get_topics())
    assert 4 == sum(1 for _ in ds.get_corpus_iter())


def test_ms_marco_dataset() -> None:
    register_to_ir_datasets("msmarco-passage/trec-dl-2019/judged")
    ds = load_dataset_with_pyterrier("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")

    assert 43 == len(ds.get_topics())
    assert 32123 == sum(1 for _ in ds.get_corpus_iter())
    assert 9260 == len(ds.get_qrels())
