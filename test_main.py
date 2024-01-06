import pytest
import time
from main import get_processing_time, clear_text
from huggingface_api import get_top_three_models
from streamlit.testing.v1 import AppTest


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


def test_get_processing_time():
    start_time = time.time()
    time.sleep(1)  # Simulate some processing time
    result = get_processing_time(start_time)
    assert isinstance(result, float), "Should return a float"
    assert result >= 1.0, "Should return the correct processing time"


def test_get_title():
    at = AppTest.from_file("main.py", default_timeout=60).run()
    assert "Text summarization" in at.title[0].value


def test_model_id_in_result_label():
    model_id_list = get_top_three_models()
    at = AppTest.from_file("main.py", default_timeout=60).run()
    at.text_area(key="text").set_value("test").run()
    at.columns[0].button(key='Summarize').click().run()
    assert model_id_list[0] in at.main.text_area(key='result 1').label
    assert model_id_list[1] in at.main.text_area(key='result 2').label
    assert model_id_list[2] in at.main.text_area(key='result 3').label


def test_get_result_value():
    at = AppTest.from_file("main.py", default_timeout=60).run()
    at.text_area(key="text").set_value("test test test test test test").run()
    at.columns[0].button(key='Summarize').click().run()
    assert 'test' in at.main.text_area(key='result 1').value
    assert 'test' in at.main.text_area(key='result 2').value
    assert 'test' in at.main.text_area(key='result 3').value
