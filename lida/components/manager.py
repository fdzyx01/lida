# Visualization manager class that handles the visualization of the data with the following methods

# summarize data given a df
# generate goals given a summary
# generate generate visualization specifications given a summary and a goal
# execute the specification given some data

import os
from typing import List, Union
import logging

import pandas as pd
from llmx import llm, TextGenerator
from lida.datamodel import Goal, Summary, TextGenerationConfig, Persona, ChartExecutorResponse, DescribeData
from lida.utils import read_dataframe
from .viz.vizconductor import VizConductor
from ..components.summarizer import Summarizer
from ..components.goal import GoalExplorer
from ..components.persona import PersonaExplorer
from ..components.executor import ChartExecutor
from ..components.viz import VizGenerator, VizEditor, VizExplainer, VizEvaluator, VizRepairer, VizRecommender

import lida.web as lida

logger = logging.getLogger("lida")


class Manager(object):
    def __init__(self, text_gen: TextGenerator = None) -> None:
        """
        Initialize the Manager object.

        Args:
            text_gen (TextGenerator, optional): Text generator object. Defaults to None.
        """

        self.text_gen = text_gen or llm()

        self.summarizer = Summarizer()
        self.goal = GoalExplorer()
        self.vizgen = VizGenerator()
        self.vizeditor = VizEditor()
        self.executor = ChartExecutor()
        self.explainer = VizExplainer()
        self.evaluator = VizEvaluator()
        self.repairer = VizRepairer()
        self.recommender = VizRecommender()
        self.conductor = VizConductor()
        self.data = None
        self.infographer = None
        self.persona = PersonaExplorer()

    def conclusion(self, code,
                   goal: Goal,
                   summary: Summary,
                   hint,
                   library,
                   textgen_config: TextGenerationConfig = TextGenerationConfig()
                   ):
        self.check_textgen(config=textgen_config)
        print("execute")
        SVG = False

        charts = self.execute(
            code_specs=[code],
            data=None,
            summary=summary,
            library=library,
            return_error=False,
            export_svg=SVG
        )
        # print(f"execute {charts}")
        if len(charts) == 1 and charts[0].status:
            chart_svg = charts[0].raster
            print("base64")
            code_specs = self.conductor.generate(
                code=code,
                hint=hint,
                goal=goal,
                summary=summary,
                svg=chart_svg,
                textgen_config=textgen_config,
                text_gen=self.text_gen,
                b64img=not SVG
            )
            return code_specs
        return []

    def check_textgen(self, config: TextGenerationConfig):
        """
        Check if self.text_gen is the same as the config passed in. If not, update self.text_gen.

        Args:
            config (TextGenerationConfig): Text generation configuration.
        """
        if config.provider is None:
            config.provider = self.text_gen.provider or "openai"
            logger.info("Provider is not set, using default provider - %s", config.provider)
            return

        if self.text_gen.provider != config.provider:
            logger.info(
                "Switching Text Generator Provider from %s to %s",
                self.text_gen.provider,
                config.provider)
            self.text_gen = llm(provider=config.provider)

    def summarize(
            self,
            data: Union[pd.DataFrame, str],
            file_name="",
            n_samples: int = 3,
            summary_method: str = "default",
            summary_hint: DescribeData = DescribeData(),
            textgen_config: TextGenerationConfig = TextGenerationConfig(n=1, temperature=0),
    ) -> Summary:
        """
        Summarize data given a DataFrame or file path.

        Args:
            data (Union[pd.DataFrame, str]): Input data, either a DataFrame or file path.
            file_name (str, optional): Name of the file if data is loaded from a file path. Defaults to "".
            n_samples (int, optional): Number of summary samples to generate. Defaults to 3.
            summary_method (str, optional): Summary method to use. Defaults to "default".
            textgen_config (TextGenerationConfig, optional): Text generation configuration. Defaults to TextGenerationConfig(n=1, temperature=0).

        Returns:
            Summary: Summary object containing the generated summary.

        Example of Summary:

            {'name': 'cars.csv',
            'file_name': 'cars.csv',
            'dataset_description': '',
            'fields': [{'column': 'Name',
            'properties': {'dtype': 'string',
                'samples': ['Nissan Altima S 4dr',
                'Mercury Marauder 4dr',
                'Toyota Prius 4dr (gas/electric)'],
                'num_unique_values': 385,
                'semantic_type': '',
                'description': ''}},
            {'column': 'Type',
            'properties': {'dtype': 'category',
                'samples': ['SUV', 'Minivan', 'Sports Car'],
                'num_unique_values': 5,
                'semantic_type': '',
                'description': ''}},
            {'column': 'AWD',
            'properties': {'dtype': 'number',
                'std': 0,
                'min': 0,
                'max': 1,
                'samples': [1, 0],
                'num_unique_values': 2,
                'semantic_type': '',
                'description': ''}},
            }

        """
        self.check_textgen(config=textgen_config)

        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data)

        self.data = data
        return self.summarizer.summarize(
            data=self.data, text_gen=self.text_gen, file_name=file_name, n_samples=n_samples,
            summary_method=summary_method, summary_hint=summary_hint, textgen_config=textgen_config)

    def goals(
            self,
            summary: Summary,
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            n: int = 5,
            persona: Persona = None,
            hint: str = ""
    ) -> List[Goal]:
        """
        Generate goals based on a summary.

        Args:
            summary (Summary): Input summary.
            textgen_config (TextGenerationConfig, optional): Text generation configuration. Defaults to TextGenerationConfig().
            n (int, optional): Number of goals to generate. Defaults to 5.
            persona (Persona, str, dict, optional): Persona information. Defaults to None.

        Returns:
            List[Goal]: List of generated goals.

        Example of list of goals:

            Goal 0
            Question: What is the distribution of Retail_Price?

            Visualization: histogram of Retail_Price

            Rationale: This tells about the spread of prices of cars in the dataset.

            Goal 1
            Question: What is the distribution of Horsepower_HP_?

            Visualization: box plot of Horsepower_HP_

            Rationale: This tells about the distribution of horsepower of cars in the dataset.
        """
        self.check_textgen(config=textgen_config)

        if isinstance(persona, dict):
            persona = Persona(**persona)
        if isinstance(persona, str):
            persona = Persona(persona=persona, rationale="")

        if isinstance(persona, Persona) and hint:
            persona.extra_hint_interest = hint

        return self.goal.generate(summary=summary, text_gen=self.text_gen,
                                  textgen_config=textgen_config, n=n, persona=persona, hint=hint)

    def personas(
            self, summary, textgen_config: TextGenerationConfig = TextGenerationConfig(),
            n=5):
        self.check_textgen(config=textgen_config)

        return self.persona.generate(summary=summary, text_gen=self.text_gen,
                                     textgen_config=textgen_config, n=n)

    def visualize(
            self,
            summary,
            goal,
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            library="seaborn",
            return_error: bool = False,
    ):
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str):
            goal = Goal(question=goal, visualization=goal, rationale="")

        self.check_textgen(config=textgen_config)
        code_specs = self.vizgen.generate(
            summary=summary, goal=goal, textgen_config=textgen_config, text_gen=self.text_gen,
            library=library)
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def execute(
            self,
            code_specs,
            data,
            summary: Summary,
            library: str = "seaborn",
            return_error: bool = False,
            export_svg = False
    ) -> List[ChartExecutorResponse]:

        if data is None:
            root_file_path = os.path.dirname(os.path.abspath(lida.__file__))
            print(root_file_path)
            data = read_dataframe(
                os.path.join(root_file_path, "files/data", summary.file_name)
            )

        # col_properties = summary.properties

        return self.executor.execute(
            code_specs=code_specs,
            data=data,
            summary=summary,
            library=library,
            return_error=return_error,
            export_svg=export_svg
        )

    def edit(
            self,
            code,
            summary: Summary,
            instructions: List[str],
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            library: str = "seaborn",
            return_error: bool = False,
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        if isinstance(instructions, str):
            instructions = [instructions]

        code_specs = self.vizeditor.generate(
            code=code,
            summary=summary,
            instructions=instructions,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def repair(
            self,
            code,
            goal: Goal,
            summary: Summary,
            feedback,
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            library: str = "seaborn",
            return_error: bool = False,
    ):
        """ Repair a visulization given some feedback"""
        self.check_textgen(config=textgen_config)
        code_specs = self.repairer.generate(
            code=code,
            feedback=feedback,
            goal=goal,
            summary=summary,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def explain(
            self,
            code,
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            library: str = "seaborn",
    ):
        """Explain a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """
        self.check_textgen(config=textgen_config)
        return self.explainer.generate(
            code=code,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def evaluate(
            self,
            code,
            goal: Goal,
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            library: str = "seaborn",
    ):
        """Evaluate a visualization code given a goal

        Args:
            code (_type_): _description_
            goal (Goal): A visualization goal

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        return self.evaluator.generate(
            code=code,
            goal=goal,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def recommend(
            self,
            code,
            summary: Summary,
            n=4,
            textgen_config: TextGenerationConfig = TextGenerationConfig(),
            library: str = "seaborn",
            return_error: bool = False,
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        code_specs = self.recommender.generate(
            code=code,
            summary=summary,
            n=n,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def infographics(self, visualization: str, n: int = 1,
                     style_prompt: Union[str, List[str]] = "",
                     return_pil: bool = False
                     ):
        """
        Generate infographics using the peacasso package.

        Args:
            visualization (str): A visualization code
            n (int, optional): The number of infographics to generate. Defaults to 1.
            style_prompt (Union[str, List[str]], optional): A style prompt or list of style prompts. Defaults to "".

        Raises:
            ImportError: If the peacasso package is not installed.
        """

        try:
            import peacasso

        except ImportError as exc:
            raise ImportError(
                'Please install lida with infographics support. pip install lida[infographics]. You will also need a GPU runtime.'
            ) from exc

        from ..components.infographer import Infographer

        if self.infographer is None:
            logger.info("Initializing Infographer")
            self.infographer = Infographer()
        return self.infographer.generate(
            visualization=visualization, n=n, style_prompt=style_prompt, return_pil=return_pil)
