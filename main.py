from huggingface_api import get_top_three_models
import streamlit as st
from transformers import pipeline, AutoTokenizer
import time

# Получить id топ-3 моделей для работы с текстом
model_id_list = get_top_three_models()


def get_processing_time(start_time):
    processing_time = round(time.time() - start_time, 2)
    return processing_time


def clear_text():
    st.session_state["text"] = ""


st.title("Text summarization")
with st.container(border=True):
    text_input = st.text_area(label='Enter some text', key="text")
    with st.container():
        col1, col2 = st.columns([0.89, 0.11])
        with col1:
            summarize_button = st.button(label='Summarize', key='Summarize')
        with col2:
            clear_button = st.button(label='Clear', key='Clear', on_click=clear_text)

if summarize_button:
    # Показываем спиннер во время обработки данных
    with st.spinner("Text summarization..."):
        i = 1
        for model_id in model_id_list:
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            summarizer = pipeline("summarization", model=model_id, tokenizer=tokenizer, use_fast=True)
            short_text = summarizer(text_input, max_length=130, min_length=14, do_sample=False)
            with st.container():
                st.text_area(
                    f"Result of summarization using [{model_id}](https://huggingface.co/{model_id}). "
                    f"Processed in {get_processing_time(start_time=start_time)} second.",
                    short_text[0]['summary_text'], key=f"result {i}")
                i += 1
        st.success("Done!")
if clear_button:
    st.rerun()
