import streamlit as st
from pdf2image import convert_from_path
import tempfile
from pathlib import Path

@st.cache_data
def process_and_index_pdfs(_retrieval_model, uploaded_files, index_name):
    if not uploaded_files:
        return {}, []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        doc_names = []

        for uploaded_file in uploaded_files:
            pdf_path = temp_path / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            doc_names.append(uploaded_file.name)

        sorted_doc_names = sorted(doc_names)

        with st.spinner(f"Indexing {len(sorted_doc_names)} PDF(s)..."):
            _retrieval_model.index(
                input_path=str(temp_path),
                index_name=index_name,
                store_collection_with_index=False,
                overwrite=True
            )

        with st.spinner("Converting PDFs to images..."):
            all_images = {}
            for doc_id, doc_name in enumerate(sorted_doc_names):
                pdf_path = temp_path / doc_name
                try:
                    images = convert_from_path(str(pdf_path))
                    all_images[doc_id] = images
                except Exception as e:
                    st.error(f"Error converting {doc_name}: {e}")
                    continue

    st.success(f"Successfully indexed and processed {len(all_images)} documents.")
    return all_images, sorted_doc_names
