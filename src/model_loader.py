import streamlit as st
import torch

from byaldi import RAGMultiModalModel
from rerankers import Reranker
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
)


@st.cache_resource
def load_models():
    with st.spinner("Loading Document Retrieval Model (ColQwen2)…"):
        device_retrieval = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            docs_retrieval_model = RAGMultiModalModel.from_pretrained(
                "vidore/colqwen2-v1.0-merged", device=device_retrieval
            )
        except Exception:
            # Fallback to CPU if the device argument is not supported
            docs_retrieval_model = RAGMultiModalModel.from_pretrained(
                "vidore/colqwen2-v1.0-merged"
            )

    with st.spinner("Loading MonoVLM reranker…"):
        device_rerank = "cuda" if torch.cuda.is_available() else "cpu"
        ranker = Reranker("monovlm", device=device_rerank)

    with st.spinner("Loading Vision-Language Model (Qwen2-VL)…"):
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
        else:
            st.warning(
                "CUDA is not available. Falling back to CPU. Inference will be slow."
            )
            vl_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id)

        vl_processor = Qwen2VLProcessor.from_pretrained(
            model_id,
            min_pixels=224 * 224,
            max_pixels=448 * 448,
        )

        vl_model.eval()

    return docs_retrieval_model, ranker, vl_model, vl_processor
