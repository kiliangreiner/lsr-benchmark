import unittest
import numpy as np
from numpy_exhaustive_search import retrieve


class TestRetrieve(unittest.TestCase):

    def test_01_gleiche_vektoren_topresult(self):
        """Identischer Vektor soll als erstes Ergebnis kommen."""
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

    def test_02_topk_wird_eingehalten(self):
        """Retrieve gibt maximal k Ergebnisse zurück."""
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

    def test_03_scores_absteigend_sortiert(self):
        """Ergebnisse sind nach Score absteigend sortiert."""
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

    def test_04_query_id_korrekt(self):
        """Query-ID wird korrekt in Ergebnisse übernommen."""
        query_ids = ["meine-query"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["d1"]
        doc_embeddings = np.array([[1.0, 0.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertEqual(results[0][0][0], "meine-query")

    def test_05_doc_id_korrekt(self):
        """Doc-ID des besten Dokuments wird korrekt zurückgegeben."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["bestes-doc", "schlechtes-doc"]
        doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=2)
        self.assertEqual(results[0][0][2], "bestes-doc")

    def test_06_mehrere_queries(self):
        """Mehrere Queries werden alle korrekt verarbeitet."""
        query_ids = ["q1", "q2"]
        query_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        doc_ids = ["d1", "d2"]
        doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0][2], "d1")
        self.assertEqual(results[1][0][2], "d2")

    def test_07_score_identischer_vektoren_ist_1(self):
        """Score von identischen normalisierten Vektoren ist 1.0."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0, 0.0]])
        doc_ids = ["d1"]
        doc_embeddings = np.array([[1.0, 0.0, 0.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertAlmostEqual(results[0][0][1], 1.0, places=5)

    def test_08_score_orthogonaler_vektoren_ist_0(self):
        """Score von orthogonalen Vektoren ist 0 – taucht nicht in Ergebnissen auf."""
        query_ids = ["q1"]
        query_embeddings = np.array([[1.0, 0.0]])
        doc_ids = ["d1"]
        doc_embeddings = np.array([[0.0, 1.0]])
        results = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, k=1)
        self.assertEqual(len(results[0]), 0)


if __name__ == "__main__":
    unittest.main()
