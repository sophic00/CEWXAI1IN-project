import streamlit as st
from pdf2image import convert_from_path
import tempfile
from pathlib import Path
from PIL import Image

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

        with st.spinner("Converting PDFs to 448×448 images and indexing..."):
            images_dir = temp_path / "images"
            images_dir.mkdir(exist_ok=True)

            all_images = {}

            for doc_id, doc_name in enumerate(sorted_doc_names):
                pdf_path = temp_path / doc_name
                try:
                    pages = convert_from_path(str(pdf_path))
                except Exception as e:
                    st.error(f"Error converting {doc_name}: {e}")
                    continue

                resized_pages = []
                for page_idx, page in enumerate(pages, start=1):
                    if page.mode != "RGB":
                        page = page.convert("RGB")

                    page.thumbnail((448, 448), Image.Resampling.LANCZOS)

                    out_path = images_dir / f"{doc_name}_page_{page_idx}.png"
                    page.save(out_path, format="PNG")

                    resized_pages.append(page)

                all_images[doc_id] = resized_pages

            _retrieval_model.index(
                input_path=str(images_dir),
                index_name=index_name,
                store_collection_with_index=False,
                overwrite=True,
            )

    st.success(f"Successfully indexed and processed {len(all_images)} document(s).")
    return all_images, sorted_doc_names
