import pytest
import time
from main import load_model, get_processing_time, shorten_text, clear_text
from huggingface_api import get_top_three_models
from transformers import AutoTokenizer, SummarizationPipeline


def test_get_top_three_models():
    result = get_top_three_models()
    print(result)
    assert result is not None
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 3, "Should return top 3 models"
    assert all(isinstance(model, str) for model in result), "Should return a list of strings"


def test_get_top_three_models_using_non_existent_pipline_tag():
    with pytest.raises(IndexError):
        get_top_three_models(pipeline_tag='qwerty')


def test_load_model():
    model_id = "t5-base"
    result = load_model(model_id)
    print(type(result))
    assert isinstance(result, SummarizationPipeline), "Should return a pipeline object"


def test_get_processing_time():
    start_time = time.time()
    time.sleep(1)  # Simulate some processing time
    result = get_processing_time(start_time)
    assert isinstance(result, float), "Should return a float"
    assert result >= 1.0, "Should return the correct processing time"


def test_shorten_text():
    original_text = "This is a long piece of text that needs to be shortened."
    model_id = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    result = shorten_text(original_text, model_id, tokenizer)[0]['summary_text']
    assert isinstance(result, str), "Should return a string"
