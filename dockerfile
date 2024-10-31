#Usar una imagen base de Python
FROM python:3.11-slim

RUN apt-get update && apt-get install -y nano

#Establecer el directorio de trabajo en /app
WORKDIR /app

# Copiar el requirements.txt e instalar dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# Copiar el contenido de la carpeta src y el archivo CSV al contenedor

COPY src/archive/ ./archive/

# Declarar el volumen (opcional si necesitas persistencia de datos)
VOLUME ./app/src

CMD ["python","src/infinite_loop.py"]