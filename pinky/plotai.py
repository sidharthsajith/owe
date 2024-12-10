import pandas as pd
import numpy as np
from datetime import datetime
import os
from pinky.prompt.prompt import Prompt
from pinky.llm.groq import GroqLLM
from pinky.code.executor import Executor
from pinky.code.logger import Logger


class PlotAI:

    def __init__(self, *args, **kwargs):
        self.model_version = "mixtral-8x7b-32768"
        self.df, self.x, self.y, self.z = None, None, None, None
        self.verbose = True
        self.output_dir = kwargs.get('output_dir', 'generated_plots')
        self.latest_plot = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for expected_k in ["x", "y", "z", "df", "model_version", "verbose"]:
            if expected_k in kwargs:
                setattr(self, expected_k, kwargs[expected_k])
        
        if self.df is None:
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    self.df = arg
                    break

    def make(self, prompt):
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        p = Prompt(prompt, self.df, self.x, self.y, self.z, save_path=filepath)
        
        if self.verbose:
            Logger().log({"title": "Prompt", "details": p.value})

        response = GroqLLM(model=self.model_version).chat(p.value)
        
        if self.verbose:
            Logger().log({"title": "Response", "details": response})

        executor = Executor()
        error = executor.run(response, globals(), {
            "df": self.df, 
            "x": self.x, 
            "y": self.y, 
            "z": self.z,
            "save_path": filepath
        })
        
        if error is not None:
            Logger().log({"title": "Error in code execution", "details": error})
            return None
        
        self.latest_plot = filepath
        return filepath

    @property
    def last_generated_plot(self):
        """Returns the path of the last generated plot"""
        return self.latest_plot