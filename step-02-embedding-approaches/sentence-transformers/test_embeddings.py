import unittest
from sentence_transformers_embeddings import embedd_text_with_model
from tempfile import TemporaryDirectory
from lsr_benchmark.irds import embeddings


class TestEmbeddings(unittest.TestCase):
    def test_01(self):
        with TemporaryDirectory() as d:
            embedding_dir = Path(d)/ "doc-embeddings.npz"
            texts = ['brown fox', 'green cat', 'yellow rabbit']
            ids = ['id-1', 'id-2','id-3']
            model = SentenceTransformer('BAAI/bge-m3')
            embedd_text_with_model(texts, ids, model, embedding_dir)
            target_embeddings = model.embedd(texts)
            actual_embeddings = embeddings(None, d, doc)
            actual_similarity = calculate_similarity(target_embeddings, actual_embeddings)
            
            self.assertEqual(len(actual_similarity), 3)
            self.assertEqual(actual_similarity[0], 1)
            self.assertEqual(actual_similarity[1], 1)
            self.assertEqual(actual_similarity[2], 1)
