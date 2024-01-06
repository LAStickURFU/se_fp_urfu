from huggingface_api import get_top_three_models
import streamlit as st
from transformers import pipeline, AutoTokenizer
import time


@st.cache_resource
def initialization():
    model_list = get_top_three_models()
    for model in model_list:
        load_model(model_id=model)


@st.cache_resource
def load_model(model_id: str):
    model = pipeline("summarization", model=model_id, from_pt=True)
    return model

initialization()

def get_processing_time(start_time):
    processing_time = round(time.time() - start_time, 2)
    return processing_time


def shorten_text(original_text: str, model_id: str, tokenizer) -> str:
    summarizer = pipeline("summarization", model=model_id)
    short_text = summarizer(original_text, max_length=130, min_length=14, do_sample=False)
    return short_text


def clear_text():
    st.session_state["text"] = ""


st.title("Text summarization")
with st.container(border=True):
    text_input = st.text_area(label='Enter some text', key="text")
    with st.container():
        col1, col2 = st.columns([0.89, 0.11])
        with col1:
            summarize_button = st.button(label='Summarize')
        with col2:
            clear_button = st.button(label='Clear', on_click=clear_text)

if summarize_button:
    # Показываем спиннер во время обработки данных
    with st.spinner("Text summarization..."):
        model_list = get_top_three_models()
        for model in model_list:
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model)
            short_text = shorten_text(original_text=text_input, model_id=model, tokenizer=tokenizer)
            with st.container():
                st.text_area(
                    f"Result of summarization using [{model}](https://huggingface.co/{model}). "
                    f"Processed in {get_processing_time(start_time=start_time)} second.",
                    short_text[0]['summary_text'])
        st.success("Done!")
if clear_button:
    st.rerun()


