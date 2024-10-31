from load_data import load_data
from main_semantic import EmbeddingSearch

# Función principal
def main(query):
    # Cargar el archivo CSV
    file_path = './src/archive/IMDB_top_1000.csv'
    df = load_data(file_path)
    
    if df is None:
        print("No se pudo cargar el archivo. Verifica la ruta y el formato del archivo CSV.")
        return
    
    search_engine = EmbeddingSearch()
    
    # Buscar coincidencias exactas en los títulos
    matching_titles = df[df['Title'].str.contains(query, case=False, na=False)]
    
    if not matching_titles.empty:
        print("Películas encontradas con el título:")
        # Eliminar duplicados basados en el título
        matching_titles = matching_titles.drop_duplicates(subset=['Title'])
        print(matching_titles[['Title', 'Description']])
    else:
        print("No se encontraron coincidencias exactas en títulos. Buscando por descripciones...")
        
        # Búsqueda semántica
        results = search_engine.search(df, query, top_n=5, context_columns=['Title', 'Description'])
        print("Películas más similares a tu búsqueda:")
        print(results[['Title', 'Description', 'similarity']])

# Ejecución del programa en un bucle infinito
if __name__ == '__main__':
    while True:
        query = input('Ingresa el término de búsqueda (o escribe "salir" para terminar): ')
        
        if query.lower() == "salir":
            print("Saliendo del programa. ¡Hasta luego!")
            break
        
        main(query)
        continuar = input("\n¿Quieres realizar otra búsqueda? (si/no): ").strip().lower()
        if continuar != "si":
            print("Saliendo del programa. ¡Hasta luego!")
            break

