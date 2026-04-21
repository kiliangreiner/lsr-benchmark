import unittest
import numpy as np
from numpy_exhaustive_search import retrieve


class TestRetrieve(unittest.TestCase):

    def test_01_identical_vector_is_top_result(self):
        """Identical vector should be returned as the top result."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0, 0.0]])
        doc_ids = ["d1", "d2", "d3"]
        doc_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=3)
        self.assertEqual(results[0][0][2], "d1")

    def test_02_topk_is_respected(self):
        """Retrieve should return at most k results."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["d1", "d2", "d3", "d4"]
        doc_embeddings = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
        ])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=2)
        self.assertLessEqual(len(results[0]), 2)

    def test_03_scores_sorted_descending(self):
        """Results should be sorted by score in descending order."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["d1", "d2", "d3"]
        doc_embeddings = np.array([
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=3)
        scores = [r[1] for r in results[0]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_04_query_id_correct(self):
        """Query ID should be correctly included in results."""
        query_ids = ["my-query"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["d1"]
        doc_embeddings = np.array([[1.0, 0.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertEqual(results[0][0][0], "my-query")

    def test_05_doc_id_correct(self):
        """Doc ID of the best document should be correctly returned."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["best-doc", "worst-doc"]
        doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=2)
        self.assertEqual(results[0][0][2], "best-doc")

    def test_06_multiple_queries(self):
        """Multiple queries should all be processed correctly."""
        query_ids = ["q1", "q2"]
        query_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        doc_ids = ["d1", "d2"]
        doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0][2], "d1")
        self.assertEqual(results[1][0][2], "d2")

    def test_07_score_of_identical_vectors_is_1(self):
        """Score of identical normalized vectors should be 1.0."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0, 0.0]])
        doc_ids = ["d1"]
        doc_embeddings = np.array([[1.0, 0.0, 0.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertAlmostEqual(results[0][0][1], 1.0, places=5)

    def test_08_score_of_orthogonal_vectors_is_0(self):
        """Score of orthogonal vectors should be 0 and not appear in results."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["d1"]
        doc_embeddings = np.array([[0.0, 1.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertEqual(len(results[0]), 0)


if __name__ == "__main__":
    unittest.main()
