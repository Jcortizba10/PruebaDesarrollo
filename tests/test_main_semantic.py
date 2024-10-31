import unittest
import pandas as pd
from unittest.mock import patch
import sys
import os
import unittest
from unittest.mock import patch

# Asegúrate de que Python pueda encontrar el directorio 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.load_data import load_data  # Cambia esta línea
from src.main_semantic import EmbeddingSearch



class TestEmbeddingSearch(unittest.TestCase):

    def setUp(self):
        # Crear un DataFrame de prueba
        data = {
            'Title': ['matrix', 'star wars', 'toy story'],
            'Description': ['The early life and career of Vito Corleone', 'Two detectives, a rookie and a veteran', 'A former Prohibition-era Jewish gangster returns']
        }
        self.df = pd.DataFrame(data)
        self.search_engine = EmbeddingSearch()

    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_create_embeddings(self, mock_encode):
        # Simular embeddings generados
        mock_encode.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        
        # Llamar al método
        result_df = self.search_engine.create_embeddings(self.df, context_columns=['Description'])
        
        # Verificar que se generaron los embeddings
        self.assertIn('embeddings', result_df.columns)
        self.assertEqual(len(result_df['embeddings']), 3)
        mock_encode.assert_called_once()

    def test_compare_similarity(self):
        # Simular embeddings para una fila
        example = {'embeddings': [0.1, 0.2]}
        query_embedding = [0.1, 0.2]

        # Llamar al método
        similarity = self.search_engine.compare_similarity(example, query_embedding)
        
        # Verificar que la similitud es correcta
        self.assertGreaterEqual(similarity, 0.99)  # Debería estar cerca de 1.0 para vectores similares

    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_search(self, mock_encode):
        # Simular embeddings
        mock_encode.side_effect = [
            [[0.1, 0.2]],  # Para el query
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Para las descripciones
        ]

        # Llamar al método search
        result_df = self.search_engine.search(self.df, 'great', top_n=2, context_columns=['Title', 'Description'])
        
        # Verificar que las columnas necesarias están en el DataFrame
        self.assertIn('similarity', result_df.columns)
        self.assertEqual(len(result_df), 2)
        mock_encode.assert_called()

    def test_load_data(self):
        # Probar cuando el archivo no existe
        with patch('os.path.exists', return_value=False):
            result = load_data('fake_path.csv')
            self.assertIsNone(result)

        # Probar cuando el archivo existe pero tiene un error al cargar
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv', side_effect=Exception('Error')):
                result = load_data('fake_path.csv')
                self.assertIsNone(result)

        # Probar cuando el archivo se carga correctamente
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv', return_value=self.df):
                result = load_data('fake_path.csv')
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 3)
    def test_create_embeddings_with_empty_df(self):
        empty_df = pd.DataFrame({'Title': [], 'Description': []})
        result_df = self.search_engine.create_embeddings(empty_df)
        self.assertTrue(result_df.empty)

    def test_compare_similarity_with_missing_embeddings(self):
        example = {}  # No embeddings
        query_embedding = [0.1, 0.2]
        similarity = self.search_engine.compare_similarity(example, query_embedding)
        self.assertEqual(similarity, 0.0)

    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_search_with_top_n_greater_than_results(self, mock_encode):
        mock_encode.side_effect = [
            [[0.1, 0.2]],  # Para el query
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Solo 3 resultados
        ]
        result_df = self.search_engine.search(self.df, 'matrix', top_n=5)
        self.assertEqual(len(result_df), 3)  # Solo debe devolver los resultados disponibles

    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_search_with_top_n_zero(self, mock_encode):
        mock_encode.side_effect = [
            [[0.1, 0.2]],  # Para el query
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Para las descripciones
        ]
        result_df = self.search_engine.search(self.df, 'matrix', top_n=0)
        self.assertEqual(len(result_df), 0)  # No debe devolver resultados cuando top_n es 0


if __name__ == '__main__':
    unittest.main()
