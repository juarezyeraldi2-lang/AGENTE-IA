import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Agente de Consulta Documental", page_icon="🤖")
st.title("🤖 Agente RAG - Consulta de Documentos (Gemini)")

st.markdown("""
Esta aplicación permite cargar un documento (**PDF** o **CSV**) y realizar preguntas sobre su contenido utilizando Inteligencia Artificial **gratuita** de Google Gemini.
""")

# API Key de Google Gemini (ingresada por el usuario)
api_key = st.sidebar.text_input("Ingresa tu Google Gemini API Key:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Sube un archivo PDF o CSV para analizar", type=["pdf", "csv"])

    if uploaded_file:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Procesando documento e indexando información...")

        try:
            # Cargar documento según extensión
            if uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)

            documents = loader.load()

            # Vectorización e indexación con Embeddings gratuitos de Google
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()

            # Configurar LLM gratuito de Gemini
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0
            )

            # Plantilla de prompt
            system_prompt = (
                "Eres un asistente para preguntas y respuestas. "
                "Usa los siguientes fragmentos de contexto para responder la pregunta de forma clara.\n"
                "Si no sabes la respuesta o el documento no la contiene, di explícitamente que no la sabes.\n\n"
                "Contexto:\n{context}\n\n"
                "Pregunta: {question}"
            )
            prompt = ChatPromptTemplate.from_template(system_prompt)

            # Función auxiliar para formatear los documentos recuperados
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Cadena RAG construida con LCEL (sin requerir langchain.chains)
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            st.success("✅ Documento listo para consultas con Gemini.")

            # Preguntas al agente
            user_query = st.text_input("Escribe tu pregunta sobre el documento:")

            if user_query:
                with st.spinner("Buscando respuesta..."):
                    response = rag_chain.invoke(user_query)
                    st.subheader("Respuesta del Agente:")
                    st.write(response)

        except Exception as e:
            st.error(f"❌ Error al procesar el documento: {e}")
else:
    st.warning("Por favor, ingresa tu Gemini API Key en la barra lateral para continuar.")
