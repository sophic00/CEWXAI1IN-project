import streamlit as st
import torch
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

@st.cache_resource
def load_models():
    with st.spinner("Loading Document Retrieval Model (ColPali)... This may take a moment."):
        device_retrieval = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            docs_retrieval_model = RAGMultiModalModel.from_pretrained(
                "vidore/colpali-v1.3-merged", device=device_retrieval
            )
        except TypeError:
            docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.3-merged")

    with st.spinner("Loading Vision Language Model (Qwen2-VL)... This is a large model."):
        if torch.cuda.is_available():
            vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            st.warning(
                "CUDA is not available. Falling back to CPU. The model may load slowly and inference \n"
                "will be significantly slower. Consider enabling a GPU Hardware upgrade in your Space if possible."
            )
            vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
                device_map={"": "cpu"}
            )

        vl_processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    return docs_retrieval_model, vl_model, vl_processor
