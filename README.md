# 🤖 Agente RAG Documental con Despliegue en OCI

Aplicación interactiva basada en Inteligencia Artificial capaz de procesar y responder preguntas sobre documentos internos (archivos PDF o CSV) mediante técnicas de RAG (_Retrieval-Augmented Generation_), desplegada en **Oracle Cloud Infrastructure (OCI)**.

---

## 📐 Arquitectura de la Solución

1. **Lectura y Procesamiento:** Uso de `PyPDFLoader` y `CSVLoader` para la ingesta de documentos estructurados y no estructurados.
2. **Vectorización y Búsqueda Semántica:** Generación de embeddings con OpenAI e indexación vectorial en memoria mediante `FAISS`.
3. **Generación de Respuestas (LLM):** Integración con LangChain y modelos GPT para responder preguntas basadas únicamente en el contexto extraído.
4. **Despliegue e Interfaz:** Interfaz web construida con **Streamlit**, alojada en una instancia Compute de **Oracle Cloud Infrastructure (OCI)**.

---

## ❓ Ejemplos de Consultas que el Agente Resuelve

- **Pregunta:** ¿Cuál fue el producto con mayor nivel de ventas según el reporte?
  - **Respuesta:** Según el archivo de ventas adjunto, el producto más vendido fue la _Licencia Cloud Enterprise_.
- **Pregunta:** ¿Qué stack de tecnologías utiliza el backend de la aplicación?
  - **Respuesta:** La documentación especifica el uso de Python con marcos de trabajo FastAPI y bases de datos PostgreSQL.

---

## ⚙️ Instrucciones para Ejecución Local

1. Clonar este repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/AGENTE_IA.git](https://github.com/TU_USUARIO/AGENTE_IA.git)
   cd AGENTE_IA
   ```
