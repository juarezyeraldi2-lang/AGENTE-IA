import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Agente de Consulta Documental", page_icon="🤖")
st.title("🤖 Agente RAG - Consulta de Documentos")

st.markdown("""
Esta aplicación permite cargar un documento (**PDF** o **CSV**) y realizar preguntas sobre su contenido utilizando Inteligencia Artificial.
""")

# API Key de OpenAI (o ingresada por el usuario)
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("Ingresa tu OpenAI API Key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Sube un archivo PDF o CSV para analizar", type=["pdf", "csv"])

    if uploaded_file:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Procesando documento e indexando información...")

        # Cargar documento según extensión
        if uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            loader = PyPDFLoader(file_path)

        documents = loader.load()

        # Vectorización e indexación en FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        # Configurar LLM, Prompt y Cadena RAG
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        system_prompt = (
            "Eres un asistente para preguntas y respuestas. "
            "Usa los siguientes fragmentos de contexto para responder la pregunta. "
            "Si no sabes la respuesta, di que no la sabes.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        st.success("✅ Documento listo para consultas.")

        # Preguntas al agente
        user_query = st.text_input("Escribe tu pregunta sobre el documento:")

        if user_query:
            with st.spinner("Buscando respuesta..."):
                response = rag_chain.invoke({"input": user_query})
                st.subheader("Respuesta del Agente:")
                st.write(response["answer"])
else:
    st.warning("Por favor, ingresa tu API Key en la barra lateral para continuar.")
