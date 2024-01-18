# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
from jinja2 import Environment, FileSystemLoader
from nbformat import v4 as nbf
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from typing import Any, Dict, List, Optional


class TextJupyterNotebook(Bolt):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def create(
        self,
        model_name: str,
        tokenizer_name: str,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        model_class: str = "AutoModelForCausalLM",
        tokenizer_class: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        torchscript: bool = False,
        compile: bool = True,
        awq_enabled: bool = False,
        flash_attention: bool = False,
        port: int = 8888,
        password: Optional[str] = None,
        **model_args: Any,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.torchscript = torchscript
        self.compile = compile
        self.awq_enabled = awq_enabled
        self.flash_attention = flash_attention
        self.model_args = model_args

        self.env = Environment(loader=FileSystemLoader("./templates"))

        # Context for Jinja template
        context = {
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "use_cuda": use_cuda,
            "precision": precision,
            "quantization": quantization,
            "device_map": device_map,
            "torchscript": torchscript,
            "compile": compile,
            "awq_enabled": awq_enabled,
            "flash_attention": flash_attention,
            "model_args": model_args,
        }
        output_path = self.output.output_folder
        class_to_template_map = {
            "AutoModelForCausalLM": "./templates/AutoModelForCausalLM.jinja",
            "AutoModelForTokenClassification": "./templates/AutoModelForTokenClassification.jinja",
            "AutoModelForSequenceClassification": "./templates/AutoModelForSequenceClassification.jinja",
            "AutoModelForTableQuestionAnswering": "./templates/AutoModelForTableQuestionAnswering.jinja",
            "AutoModelForQuestionAnswering": "./templates/AutoModelForQuestionAnswering.jinja",
            "AutoModelForSeq2SeqLM": "./templates/AutoModelForSeq2SeqLM.jinja",
        }

        template_name = class_to_template_map[model_class]

        self.create_notebook(name=template_name, context=context, output_path=f"{output_path}/notebook.ipynb")

        self.install_packages(
            [
                "numpy==1.21.6",
                "scikit-learn==1.3.0",
                "pandas==1.3.5",
                "matplotlib-inline==0.1.6",
                "seaborn==0.13.1",
                "torch==2.1.2",
                "tensorflow==2.15.0",
                "transformers",
                "datasets",
                "evaluate",
                "diffusers",
                "nemo_toolkit[all]",
                "jupyterthemes",
                "jupyter==1.0.0",
            ]
        )
        self.install_jupyter_extensions(
            [
                "jupyter_contrib_nbextensions",
                "jupyter_nbextensions_configurator",
                "jupyter_tensorboard",
                "rise",
                "nbdime",
            ]
        )
        self.enable_jupyter_dark_theme()

        self.start_jupyter_server(notebook_dir=output_path, port=port, password=password)

    def create_notebook(self, name: str, context: dict, output_path: str):
        """
        Create a Jupyter Notebook from a Jinja template.

        Args:
        context (dict): Context variables to render the template.
        output_path (str): Path to save the generated notebook.
        """
        template = self.env.get_template(name)
        notebook_json = template.render(context)
        notebook = nbf.reads(notebook_json, as_version=4)

        with open(output_path, "w") as f:
            nbf.write(notebook, f)
        self.log.info(f"Notebook created at {output_path}")

    def start_jupyter_server(self, notebook_dir: str, port: int = 8888, password: Optional[str] = None):
        """
        Start a Jupyter Notebook server in the specified directory with an optional port and password.

        Args:
        notebook_dir (str): Directory where the notebook server should start.
        port (int): Port number for the notebook server. Default is 8888.
        password (Optional[str]): Password for accessing the notebook server. If None, no password is set.
        """
        command = ["jupyter", "notebook", "--notebook-dir", notebook_dir, "--port", str(port)]

        if password:
            from notebook.auth import passwd

            hashed_password = passwd(password)
            command.extend(["--NotebookApp.password", f"'{hashed_password}'"])

        subprocess.run(command, check=True)

    def install_packages(self, packages: List[str]):
        """
        Install Python packages using pip.

        Args:
        packages (List[str]): List of package names to install.
        """
        for package in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        self.log.info("Required packages installed.")

    def install_jupyter_extensions(self, extensions: List[str]):
        """
        Install Jupyter Notebook extensions.

        Args:
        extensions (List[str]): List of Jupyter extension names to install.
        """
        for extension in extensions:
            subprocess.run(["jupyter", "nbextension", "install", extension, "--user"], check=True)
            subprocess.run(["jupyter", "nbextension", "enable", extension, "--user"], check=True)
        self.log.info("Jupyter extensions installed and enabled.")

    def enable_jupyter_dark_theme(self):
        """
        Enable dark theme for Jupyter Notebook.
        """
        subprocess.run(["jt", "-t", "onedork"], check=True)  # Example: using 'onedork' theme from jt (jupyterthemes)
        self.log.info("Jupyter dark theme enabled.")
