from test_lib import *

summary, goals = summary_and_goals("times.csv")
#%%
print(parse_goal(goals))
#%%
goal = goals[1]
visu = req(
    uri="visualize",
    data={
        "summary": summary,
        "goal": goal,
        "library": "seaborn",
        "textgen_config": text_gen_config
    }
)
#%%
visu["charts"][0]["code"]
#%%
base64_data = visu["charts"][0]["raster"]
img = Image.open(io.BytesIO(base64.b64decode(base64_data)))
img
#%%
img.close()
#%%
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
#%%
md(conclusion["charts"][0]["content"])