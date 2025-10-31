import unittest


def load_dataset_with_pyterrier(irds_id):
        import pyterrier as pt
        return pt.get_dataset(f"irds:{irds_id}")


class TestIrdsIntegrationToPyTerrier(unittest.TestCase):
    def test_that_original_dataset_can_be_loaded(self):
        dataset = load_dataset_with_pyterrier("clueweb09/en/trec-web-2009")
        self.assertIsNotNone(dataset)
