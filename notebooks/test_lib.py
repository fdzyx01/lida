import time
import markdown
import requests
import json
import base64
import io
from PIL import Image

# from IPython.display import Markdown as md

md = markdown.Markdown()

base_url = "http://127.0.0.1:8080/api/"
text_gen_config = {
    "temperature": 0,
    "n": 1,
    "model": "gpt-3.5-turbo-0301",
    "max_tokens": None,
    "provider": "openai"
}


def req(uri, data, file=None):
    # print(f"############ {uri} ##############")
    response = requests.post(
        url=base_url + uri,
        data=json.dumps(data) if file is None else {"data": json.dumps(data)},
        files=None if file is None else {"file": file},
        headers=None if file else {"Content-Type": "application/json"}
    )
    # print(response.text)
    return json.loads(response.text)


def parse_goal(goals):
    return [str(p["index"]) + ". " + p["question"] for p in goals]


def summary_and_goals(filename, hint="", description="", fields=dict()):
    # %%
    if fields is None:
        fields = {}
    summary = req(uri="summarize", data={
        "description": description,
        "fields": fields
    }, file=open(filename))["summary"]

    # %%
    goals = req(
        uri="goal",
        data={
            "extra_hint_interest": hint,
            "summary": summary,
            "n": 20,
            "textgen_config": text_gen_config
        }
    )["data"]
    return summary, goals


def visu_and_conclusion(summary, goal):
    visu = req(
        uri="visualize",
        data={
            "summary": summary,
            "goal": goal,
            "library": "seaborn",
            "textgen_config": text_gen_config
        }
    )
    print(visu)
    base64_data = visu["charts"][0]["raster"]
    conclusion = req(
        uri="visualize/conclusion",
        data={
            "summary": summary,
            "goal": goal,
            "textgen_config": {
                "temperature": 0,
                "n": 1,
                "model": "gpt-4-turbo",
                "max_tokens": None,
                "provider": "openai"
            },
            "library": "seaborn",
            "code": visu["charts"][0]["code"]
        }
    )
    # %%
    rest = md.convert(conclusion["charts"][0]["content"])
    return base64_data, rest

html_head = '''<html><body><h1 style="text-align: center;">{filename}</h1>'''
html_goals_head = '''
<h2>goal</h2>
'''
html_goals_item = '''
<p style="{style}">{goal}</p>
'''
html_back = '''</body></html>'''
html_template = '''<div>
<h2 style="text-align: left">{ques}</h2>
<p>{time}</p>
<div style="width: 1000px;display: flex;justify-content: space-between;">
<img src="data:image/jpeg;base64,{img}" width="300" height="300" />
<div style="max-height: 500px; overflow: scroll;">{md}</div>
</div>
</div>'''
def build_test_file(filename):
    f = open(f'{filename}.html', 'w')
    f.write(html_head.format(filename=filename))
    return f

def add_goals_to_file(f, goals, targets=[]):

    f.write(html_goals_head)
    for index, i in enumerate(goals):
        style = ""
        if index in targets:
            style = "color: red"
        d = html_goals_item.format(goal=i, style=style)
        f.write(d)

def add_test_file(f, ques, md, img):
    f.write(html_template.format(ques=ques, md=md, img=img, time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

def close_test_file(f):
    f.write(html_back)
    f.close()

