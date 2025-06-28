import streamlit as st
import torch
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

@st.cache_resource
def load_models():
    with st.spinner("Loading Document Retrieval Model (ColPali)... This may take a moment."):
        docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

    with st.spinner("Loading Vision Language Model (Qwen2-VL)... This is a large model."):
        if not torch.cuda.is_available():
            st.error("CUDA is not available. This application requires a GPU to run.")
            st.stop()

        vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        vl_processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    return docs_retrieval_model, vl_model, vl_processor
