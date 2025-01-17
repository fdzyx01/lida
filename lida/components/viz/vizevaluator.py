
import json
from ...utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from lida.datamodel import Goal

system_prompt = """
You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
- bugs (bugs):  Based on the errors encountered while executing the provided code, if there are no errors, assign a score of 10. If errors are present, the score must be below 5. Additionally, explain the reasons for the errors and provide detailed improvement suggestions. For example, "In seaborn's catplot, for a count plot, you should specify only the x or y parameter, not both."
- Data transformation (transformation): Is the data transformed appropriately for the visualization type? Only consider the data transformation operations during plotting, not the operations during data cleaning. E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed? If there are any issues, you need to deduct points; and you must explain the reasons for the issues and detailed improvement plans.
- Goal compliance (compliance): how well the code meets the specified visualization goals?Is it suitable for the intended audience? Does it provide an adequate amount of information, neither oversimplified nor overly complex? Does it avoid unnecessary or irrelevant information? Is the presentation and organization of data logically clear, aiding in understanding and analysis? If there are any issues, you need to deduct points; and you must explain the reasons for the issues and detailed improvement plans.
- Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST be less than 5. If there are any issues, you must deduct points; and you must explain the reasons for the errors and provide detailed improvement plans, such as which type of visualization should be used.
- Data encoding (encoding): Is the data encoded appropriately for the visualization type?Does the data encoding accurately represent the data values? Does the encoding introduce any misunderstandings or misinterpretations? Is the encoded data easy to understand? Can users easily read information from the chart? If there are any issues, you must deduct points; and you must explain the reasons for the errors and the specific improvement plans.
- aesthetics (aesthetics): Are the visualization title and axis labels in their most concise form? Do the axis labels include units if they are very certain? If any issues are found, deductions must be made; special attention is required: you must identify all issues, explain the reasons for these issues, and provide a detailed improvement plan, such as how the visualization title could be simplified, what units could be added to the axes, and you must specify how to implement these improvements. For example, the title "How does car ownership vary among individuals with different numbers of children?" is not concise, To make it more concise while retaining the main information of the original title, you could revise it to "Car Ownership by Number of Children". If there are no issues, you need to give a score of 10, but please be cautious with this action.

You must provide a score for each of the above dimensions.  Assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., matplotlib, seaborn etc object).

You need to pay special attention: you must be extremely cautious about awarding high scores. In evaluating each dimension, any issue with the details will result in points being deducted. High scores are only given when something is exceptionally perfect. You must not give high scores lightly.However, any deductions must be justified and cannot be based on mere possibilities. Additionally, you need to explain the reasons for the deductions and suggest improvements.

Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format(When you output, you must replace 'a number' after 'score' with a specific number from 0 to 10; write the reason for giving this score in the value of the rationale.):

```[
{ "dimension":  "bugs",  "score": a number , "rationale": " .."}, { "dimension":  "transformation",  "score": a number, "rationale": " .."}, { "dimension":  "compliance",  "score": a number, "rationale": " .."},{ "dimension":  "type",  "score": a number, "rationale": " .."}, { "dimension":  "encoding",  "score": a number, "rationale": " .."}, { "dimension":  "aesthetics",  "score": a number, "rationale": " .."}
]
```
"""


class VizEvaluator(object):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
    ) -> None:
        pass

    def generate(self, code: str, goal: Goal,
             textgen_config: TextGenerationConfig, text_gen: TextGenerator, error:str, library='altair'):
        """Generate a visualization explanation given some code"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant",
            "content": f"Generate an evaluation given the goal and code below in {library}. The specified goal is \n\n {goal.question} \n\n and the visualization code is \n\n {code} \n\n. The error reported by executing this code is: {error}. Now, evaluate the code based on the 6 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. You must ensure that the score in your output is a number from 0 to 10, not the string 'a number'. Please write the reason for giving this score in the value of the rationale. Your structured evaluation is below."},
        ]

        try:
            completions: TextGenerationResponse = text_gen.generate(
                messages=messages, config=textgen_config)

            completions = [clean_code_snippet(x['content']) for x in completions.text]
            evaluations = []
            for completion in completions:
                try:
                    evaluation = json.loads(completion)
                    evaluations.append(evaluation)
                except Exception as inner_json_error:
                    print("Error parsing evaluation data after modification", completion, str(inner_json_error))
        except Exception as json_error:
            # Update the assistant's message and retry text generation
            messages[1]["content"] = messages[1]["content"].replace(
                "Your structured evaluation is below.",
                "Your structured evaluation is:")
            completions: TextGenerationResponse = text_gen.generate(
                messages=messages, config=textgen_config)
            completions = [clean_code_snippet(x['content']) for x in completions.text]
            evaluations = []
            for completion in completions:
                print("retry...")
                try:
                    evaluation = json.loads(completion)
                    evaluations.append(evaluation)
                except Exception as inner_json_error:
                    print("Error parsing evaluation data after modification", completion, str(inner_json_error))

        return evaluations
