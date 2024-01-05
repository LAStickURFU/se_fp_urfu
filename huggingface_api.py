import requests

url = "https://huggingface.co/models-json"


def get_top_three_models(pipeline_tag: str = "summarization", sort: str = 'likes') -> list:
    params = {"pipeline_tag": pipeline_tag, "sort": sort}
    response = requests.request("GET", url, params=params)
    json_data = response.json()
    top_three_models = []
    for i in range(3):
        top_three_models.append(json_data["models"][i]["id"])
    return top_three_models
