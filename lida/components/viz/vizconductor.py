import json

from lida.datamodel import Goal, Summary

from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from ..scaffold import ChartScaffold

system_prompt = """
You are a helpful data-scientist highly skilled in providing helpful suggestions and explanations of visualization of the plot. You will be give a plot and its description. You should conduct some results or conclusion about the topic and its rationale and give some helpful suggestions. You MUST follow these rules:
1. Your response at least contains 2 section, and more than 500 words totally.
2. In the first section, you should give some answers which DIRECTLY solve the topic question. You do NOT describe a lot of plain or sample explanation for plot, just give the key description of plot around the topic.
3. In the second section, you should give some insight and inference about the topic based on current data. Such as what things may be caused in society if the trend of the data consisting, increasing or decreasing. And what impact does it have and how does it affect. If the impact has bad effect, you should also give some solutions to change the trend.
"""

format_instructions = """
Your output MUST be perfect JSON in THE FORM OF A VALID LIST of JSON OBJECTS WITH PROPERLY ESCAPED SPECIAL CHARACTERS e.g.,

```[
    {"section": "accessibility", "code": "None", "explanation": ".."}  , {"section": "transformation", "code": "..", "explanation": ".."}  ,  {"section": "visualization", "code": "..", "explanation": ".."}
    ] ```

The code part of the dictionary must come from the supplied code and should cover the explanation. The explanation part of the dictionary must be a string. The section part of the dictionary must be one of "accessibility", "transformation", "visualization" with no repetition. THE LIST MUST HAVE EXACTLY 3 JSON OBJECTS [{}, {}, {}].  THE GENERATED JSON  MUST BE A LIST IE START AND END WITH A SQUARE BRACKET.
"""


class VizConductor(object):
    """Generate visualizations Explanations given some code"""

    def __init__(
            self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, svg: str, hint: str, goal: Goal,
            summary: Summary,
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='seaborn', b64img: bool = False):
        """Generate a visualization explanation given some code"""
        # {goal.rationale}.
        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt
            }]
        }, {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": f"This plot is about the topic of {goal.question}, showing a {goal.visualization}." + (
                    f"The svg-formatted plot to be explained is: \n {svg}\n" if not b64img else "")
            }]
        }, {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + svg
                    }
                },
                {
                    "type": "text",
                    "text": f"Your response for the plot above is: \n\n"
                }
            ] if b64img else [
                {
                    "type": "text",
                    "text": f"Your response for the plot above is: \n\n"
                }
            ]
        }]
        print(messages)
        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)

        explanations = completions.text
        print(completions.to_dict())
        return explanations
