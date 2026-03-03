from json import load as json_load
from pathlib import Path
from typing import Iterator

from approvaltests import verify_as_json
from pytest import fixture
from spacy import cli

from lsr_benchmark.corpus.segmentation import segmented_document


def load_docs() -> dict[str, str]:
    ret = {}
    for i in ["1", "2"]:
        with (Path(__file__).parent / "resources" / f"example-dl-0{i}.json").open("rb") as file:
            ret[i] = json_load(file)["text"]
    return ret


@fixture
def setup_spacy() -> Iterator[None]:
    cli.download("en_core_web_sm")
    yield None


def test_load_docs(setup_spacy) -> None:
    docs = load_docs()
    assert docs is not None


def test_chunking(setup_spacy) -> None:
    docs = load_docs()
    actual = segmented_document(docs, 200)
    verify_as_json(actual)
