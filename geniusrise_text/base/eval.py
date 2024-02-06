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

from typing import Any, Dict, Optional

import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger

from .bulk import TextBulk


class TextEval(TextBulk):
    """
    TextEval serves as a base class for task-specific evaluation classes, facilitating the evaluation of models on
    various tasks like text classification, summarization, etc. It inherits from TextBulk, leveraging its infrastructure
    for bulk text processing while introducing functionalities specific to evaluation, such as loading models with
    evaluation configurations, handling datasets, and computing evaluation metrics.

    Attributes:
        model (Any): The pre-trained language model used for evaluation.
        tokenizer (Any): The tokenizer for preprocessing input text.
        model_name (str): The identifier of the pre-trained language model.
        model_revision (Optional[str]): The specific revision of the model, if applicable.
        tokenizer_name (str): The identifier of the tokenizer.
        tokenizer_revision (Optional[str]): The specific revision of the tokenizer, if applicable.
        model_class (str): The class name of the model, indicating its type (e.g., for sequence classification).
        tokenizer_class (str): The class name of the tokenizer.
        use_cuda (bool): Flag to enable CUDA for GPU acceleration during evaluation.
        quantization (int): Level of quantization for model optimization (0 for none).
        precision (str): Computational precision configuration (e.g., "float32", "float16").
        device_map (str | Dict | None): Device configuration for model inference.
        max_memory (Dict[int, str]): Maximum memory allocation for model inference per device.
        torchscript (bool): Flag to use TorchScript-optimized models for faster inference.
        model_args (Any): Additional arguments for model configuration.
    """

    model: Any
    tokenizer: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
    ):
        """
        Initializes the TextEval class with configurations for input, output, and model state management.

        Args:
            input (BatchInput): Configuration for input data processing.
            output (BatchOutput): Configuration for output data handling.
            state (State): State management configuration.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def load_models(
        self,
        model_name: str,
        model_class: str = "AutoModelForCausalLM",
        tokenizer_class: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = False,
        compile: bool = True,
        awq_enabled: bool = False,
        flash_attention: bool = False,
        **model_args: Any,
    ) -> None:
        """
        Loads and configures the specified model and tokenizer for evaluation, optimizing for performance based on the
        provided configurations.

        Args:
            model_name (str): Identifier for the pre-trained model to load.
            model_class (str, optional): Class of the model to be loaded. Defaults to "AutoModelForCausalLM".
            tokenizer_class (str, optional): Class of the tokenizer to be loaded. Defaults to "AutoTokenizer".
            use_cuda (bool, optional): Flag to enable GPU acceleration. Defaults to False.
            precision (str, optional): Computational precision. Defaults to "float16".
            quantization (int, optional): Level of model quantization. Defaults to 0.
            device_map (str | Dict | None, optional): Device mapping for model inference. Defaults to "auto".
            max_memory (Dict[int, str], optional): Maximum memory allocation per device. Defaults to {0: "24GB"}.
            torchscript (bool, optional): Flag to use TorchScript for model optimization. Defaults to False.
            compile (bool, optional): Whether to compile the model for performance optimization. Defaults to True.
            awq_enabled (bool): Flag to enable Adaptive Weight Quantization. Defaults to False.
            flash_attention (bool): Flag to enable Flash Attention optimization. Defaults to False.
            **model_args (Any): Additional arguments for model configuration.
        """
        self.model_name = model_name
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.use_cuda = use_cuda
        self.quantization = quantization
        self.precision = precision
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.flash_attention = flash_attention
        self.awq_enabled = awq_enabled
        self.model_args = model_args

        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            tokenizer_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            tokenizer_name = model_name
        else:
            model_revision = None
            tokenizer_revision = None
            tokenizer_name = model_name

        self.model_name = model_name
        self.model_revision = model_revision
        self.tokenizer_name = tokenizer_name
        self.tokenizer_revision = tokenizer_revision

        self.model, self.tokenizer = self.load_models(
            model_name=self.model_name,
            tokenizer_name=self.tokenizer_name,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            model_class=self.model_class,
            tokenizer_class=self.tokenizer_class,
            use_cuda=self.use_cuda,
            precision=self.precision,
            quantization=self.quantization,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            awq_enabled=self.awq_enabled,
            flash_attention=self.flash_attention,
            compile=compile,
            **self.model_args,
        )
