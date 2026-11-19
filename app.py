import streamlit as st

from rag_backend import answer_question, get_indexed_chunk_count, ingest_pdfs


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Chatbot")
st.caption("Upload PDFs and ask questions grounded in their contents.")

with st.sidebar:
    st.header("Your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        pdfs = [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files]
        with st.spinner("Reading PDFs and creating embeddings..."):
            chunk_count = ingest_pdfs(pdfs)
        st.success(f"Processed {len(pdfs)} PDF(s) into {chunk_count} text chunks.")

    st.caption(f"Indexed text chunks: {get_indexed_chunk_count()}")
    st.caption("The local index is saved in `data/chroma_db/`.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs..."):
            answer, sources = answer_question(question)
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- {source['source']}, page {source['page']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})