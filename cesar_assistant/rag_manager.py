import os
import faiss
import numpy as np
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.genai import types

class RAGManager:
    def __init__(self, client, model_name="text-embedding-004"):
        self.client = client
        self.model_name = model_name
        self.dimension = 768  # Dimensión estándar para text-embedding-004
        # HNSW con métrica L2 (Euclidiana)
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
        self.documents = []  # Almacena los trozos de texto originales
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_file(self, file_path):
        """Procesa un archivo (PDF o MD) y lo añade al índice."""
        text = ""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        elif ext == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return f"Extensión {ext} no soportada para RAG."

        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return "No se extrajo texto del archivo."

        # Generar embeddings para los chunks
        try:
            embeddings_response = self.client.models.embed_content(
                model=self.model_name,
                contents=chunks,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            # Extraer los vectores
            vectors = np.array([item.values for item in embeddings_response.embeddings]).astype('float32')
            
            # Añadir al índice FAISS
            self.index.add(vectors)
            self.documents.extend(chunks)
            return f"Procesado: {len(chunks)} fragmentos añadidos al índice."
        except Exception as e:
            return f"Error generando embeddings: {str(e)}"

    def search(self, query, top_k=5):
        """Busca los fragmentos más relevantes para una consulta."""
        if self.index.ntotal == 0:
            return []

        try:
            query_embedding_response = self.client.models.embed_content(
                model=self.model_name,
                contents=[query],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_vector = np.array([query_embedding_response.embeddings[0].values]).astype('float32')
            
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.documents):
                    results.append(self.documents[idx])
            return results
        except Exception as e:
            # Error suppressed for production; consider logging if needed
            return []

    def clear(self):
        """Limpia el índice y los documentos."""
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.documents = []
