import streamlit as st
from qwen_vl_utils import process_vision_info
import torch
import base64
from io import BytesIO

def get_grouped_images(results, all_images):
    grouped_images = []
    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        if doc_id in all_images and page_num <= len(all_images[doc_id]):
            grouped_images.append(all_images[doc_id][page_num - 1])
    return grouped_images

def images_to_base64(images):
    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(img_base64)
    return base64_images

def process_ranker_results(results, grouped_images, top_k=1):
    new_images = []
    try:
        docs = results.top_k(top_k)
    except AttributeError:
        docs = results[:top_k]

    for doc in docs:
        new_images.append(grouped_images[doc.doc_id])
    return new_images

def answer_question(
    query,
    docs_retrieval_model,
    ranker,
    vl_model,
    vl_processor,
    all_images,
    index_name,
    retrieval_top_k=3,
    reranker_top_k=1,
    max_new_tokens=500,
):
    with st.spinner("Searching for relevant pages in the document(s)…"):
        results = docs_retrieval_model.search(query, k=retrieval_top_k, index_name=index_name)
        grouped_images = get_grouped_images(results, all_images)

    if not grouped_images:
        st.warning(
            "Could not retrieve any relevant images for your query. Please try a different question."
        )
        return None, None

    with st.spinner("Reranking retrieved images…"):
        base64_imgs = images_to_base64(grouped_images)
        rerank_results = ranker.rank(query, base64_imgs)
        ranked_images = process_ranker_results(rerank_results, grouped_images, top_k=reranker_top_k)

    chat_template = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in ranked_images]
            + [{"type": "text", "text": query}],
        }
    ]

    with st.spinner("Generating an answer based on the retrieved images…"):
        process_vision_result = process_vision_info(chat_template)

        if isinstance(process_vision_result, tuple) and len(process_vision_result) == 3:
            image_inputs = process_vision_result[0]
        else:
            image_inputs = process_vision_result

        text_prompt = vl_processor.apply_chat_template(
            chat_template, tokenize=False, add_generation_prompt=True
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs = {
            "text": [text_prompt],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }

        if isinstance(process_vision_result, tuple) and len(process_vision_result) == 3:
            _, video_inputs, _ = process_vision_result
            kwargs["videos"] = video_inputs

        inputs = vl_processor(**kwargs).to(device)

        generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return (
        output_text[0] if output_text else "No answer could be generated.",
        ranked_images,
    )
