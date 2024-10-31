import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Clase para manejar la búsqueda de similitud de embeddings
class EmbeddingSearch:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def create_embeddings(self, df, context_columns=['Description']):
        """
        Genera embeddings para las columnas seleccionadas del DataFrame y las agrega como una nueva columna.
        """
        df['context'] = df[context_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        try:
            embeddings = self.model.encode(df['context'], batch_size=64, show_progress_bar=True)
            #df['embeddings'] = embeddings.tolist()
            df['embeddings'] = list(embeddings)  # Asegúrate de convertir a lista
            return df
        except Exception as e:
            print(f"Error al generar embeddings: {e}")
            return df
    
    def compare_similarity(self, example, query_embedding):
        """
        Calcula la similitud coseno entre dos embeddings.
        """
        try:
            embedding = example['embeddings']
            similarity = util.cos_sim(embedding, query_embedding).item()
            return similarity
        except Exception as e:
            print(f"Error al calcular similitud: {e}")
            return 0.0

    def search(self, df, query, top_n=5, context_columns=['Description']):
        """
        Realiza una búsqueda semántica usando embeddings y devuelve las películas más similares al término de búsqueda.
        """
        query_embedding = self.model.encode([query])[0]
        df = self.create_embeddings(df, context_columns)
        
        # Calcular similitudes
        df['similarity'] = df.apply(lambda x: self.compare_similarity(x, query_embedding), axis=1)
        
        # Eliminar duplicados basados en el título
        df = df.drop_duplicates(subset=['Title'])
        
        # Ordenar las películas por similitud
        df = df.sort_values(by='similarity', ascending=False)
        return df.head(top_n)
