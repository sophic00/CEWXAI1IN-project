import streamlit as st
from model_loader import load_models
from data_processing import process_and_index_pdfs
from rag_pipeline import answer_question

st.set_page_config(
    page_title="Multimodal RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDEX_NAME = "pdf_upload_index"

st.title("📄 Multimodal RAG with your own PDFs")
st.markdown("""
This application allows you to ask questions about your own PDF documents.
1.  **Upload PDFs**: Start by uploading one or more PDF files in the sidebar.
2.  **Ask a Question**: The system will find relevant pages and use a Vision-Language Model to answer.
""")

st.sidebar.header("Controls")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF documents",
    type="pdf",
    accept_multiple_files=True
)

docs_retrieval_model, vl_model, vl_processor = load_models()

if uploaded_files:
    all_images, doc_names = process_and_index_pdfs(docs_retrieval_model, uploaded_files, INDEX_NAME)

    st.sidebar.success("Ready to answer questions!")
    doc_list = "\n- ".join(doc_names)
    st.sidebar.info(f"**Indexed Documents:**\n- {doc_list}")

    st.header("Ask a Question")
    top_k = st.slider("Number of images to retrieve (Top-K)", min_value=1, max_value=10, value=3)
    query = st.text_input("Enter your question about the document(s):", "What is the main topic of this document?")

    if st.button("Get Answer", type="primary"):
        if query:
            answer, retrieved_images = answer_question(
                query, docs_retrieval_model, vl_model, vl_processor, all_images, INDEX_NAME, top_k=top_k
            )

            if answer and retrieved_images:
                st.header("Answer")
                st.markdown(answer)

                st.header("Retrieved Images")
                st.write(f"These {len(retrieved_images)} images were used to generate the answer:")
                cols = st.columns(len(retrieved_images))
                for i, img in enumerate(retrieved_images):
                    with cols[i]:
                        st.image(img, caption=f"Retrieved Image {i+1}", use_column_width=True)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload at least one PDF document to begin.")
