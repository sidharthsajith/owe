class Prompt:

    def __init__(self, prompt="", df=None, x=None, y=None, z=None, previous_code="", previous_error="", save_path=None, data={}):
        self.prompt = prompt
        self.df = df
        self.x, self.y, self.z = x, y, z
        self.previous_code = previous_code
        self.previous_error = previous_error
        self.save_path = save_path

    def input_data_str(self):
        if self.df is not None:
            return f"""
```python
# pandas DataFrame
'''
{self.df.head(5)}
'''
# DataFrame columns
'''
{self.df.columns.to_list()}
'''

# pandas data frame variable is df
```
            """
        else:
            pass

    @property
    def value(self):
        save_instruction = f"Save the plot to '{self.save_path}' using plt.savefig(save_path) before plt.close()" if self.save_path else ""
        
        v = f"""Create a plot in Python with matplotlib package.

Input data:

{self.input_data_str()}

Plot should contain: {self.prompt}

Initial python code to be updated        

```python
# TODO import required dependencies
# TODO Provide the plot
# TODO {save_instruction}
```

Important: Make sure to save the plot using plt.savefig(save_path) and then close it with plt.close()
Output only Python code.
"""

        if self.previous_code != "":
            v += f"""
            
            You generated previously below code:
            {self.previous_code}

            It returned below error:
            {self.previous_error}

            Fix it. Do not return the same code again.
            """
        return v