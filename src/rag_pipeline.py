import streamlit as st
from qwen_vl_utils import process_vision_info

def get_grouped_images(results, all_images):
    grouped_images = []
    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        if doc_id in all_images and page_num <= len(all_images[doc_id]):
            grouped_images.append(all_images[doc_id][page_num - 1])
    return grouped_images

def answer_question(query, docs_retrieval_model, vl_model, vl_processor, all_images, index_name, top_k=3, max_new_tokens=500):
    with st.spinner("Searching for relevant pages in the document(s)..."):
        results = docs_retrieval_model.search(query, k=top_k, index_name=index_name)
        retrieved_images = get_grouped_images(results, all_images)

    if not retrieved_images:
        st.warning("Could not retrieve any relevant images for your query. Please try a different question.")
        return None, None

    chat_template = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in retrieved_images]
            + [{"type": "text", "text": f"Based on the provided images, answer the following question: {query}"}],
        }
    ]

    with st.spinner("Generating an answer based on the retrieved images..."):
        process_vision_result = process_vision_info(chat_template)
        if isinstance(process_vision_result, tuple) and len(process_vision_result) == 3:
            image_inputs = process_vision_result[0]
        else:
            image_inputs = process_vision_result
        text = vl_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
        inputs = vl_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0] if output_text else "No answer could be generated.", retrieved_images
