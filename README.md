---
title: Cesar Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.16.2
app_file: cesar_assistant/app.py
pinned: false
license: mit
---

# Cesar Assistant: Advanced Research RAG Agent 🚀

Cesar Assistant es un agente de IA analítico y multimodal de última generación, diseñado para investigación profunda, análisis de datos y asistencia empresarial. Utiliza el stack tecnológico más avanzado para integrar memoria persistente, búsqueda vectorial y razonamiento multimodal.

## 🛠️ Tecnologías Core

- **Motor de IA**: Google Gemini (Family 1.5/2.0 Pro & Flash) vía `google-genai`.
- **RAG (Búsqueda Vectorial)**: FAISS con índice HNSW y distancia Euclidiana (L2).
- **Procesamiento de Documentos**: PyMuPDF para PDF y segmentación recursiva semántica.
- **Base de Datos y Memoria**: Supabase (PostgreSQL) con esquemas de perfiles dinámicos.
- **Visualización**: Matplotlib (Server-side Agg backend) y Pandas para análisis tabular.
- **Interfaz**: Gradio para una experiencia de usuario fluida y reactiva.

## ✨ Características Principales

- **Investigación Multimodal**: Carga y analiza simultáneamente archivos PDF, Markdown e Imágenes técnicas.
- **Memoria Persistente**: Recuperación de perfiles de usuario y conversaciones previas desde Supabase.
- **Gradientes de Temperatura Dinámicos**:
  - **Modo Analítica**: `Temp 0.0` para máxima precisión y cero alucinaciones.
  - **Modo Conversacional**: `Temp 0.7` para fluidez estratégica.
- **Perfilado de Personalidad**: Análisis NLP en tiempo real del estilo de comunicación y reglas de negocio del usuario.
- **Visualización Automática**: Generación de gráficos de barras, líneas y dispersión a partir de datos procesados por la IA.

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd cesar_assistant
```

### 2. Configurar el Entorno

Crea un archivo `.env` basado en el `.env.example`:

- `GEMINI_API_KEY`: Tu clave de Google AI Studio.
- `SUPABASE_URL` & `SUPABASE_KEY`: Credenciales de tu proyecto Supabase.
- `HF_TOKEN`: Token de Hugging Face (opcional para modelos adicionales).

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Inicializar Base de Datos (Supabase)

Ejecuta el contenido de `supabase_schema.sql` en el SQL Editor de tu Dashboard de Supabase para crear las tablas y políticas RLS necesarias.

### 5. Ejecutar la Aplicación

```bash
python app.py
```

## 📂 Estructura del Proyecto

- `app.py`: Orquestador principal y servidor Gradio.
- `rag_manager.py`: Motor de búsqueda semántica y gestión de FAISS.
- `supabase_schema.sql`: Definición de la estructura de datos y seguridad.
- `.agent.md`: Definición del agente de mantenimiento del sistema.

## 📝 Uso Sugerido para Investigación

Sube documentos técnicos y pide: *"Analiza las tendencias de rentabilidad descritas en este PDF basándote en la imagen adjunta y genera un reporte tabular con gráfico de líneas."*

---
**Desarrollado como un Research Agent inteligente para el ecosistema de Cesar Mora.**
