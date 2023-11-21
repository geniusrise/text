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

from typing import Any, Dict

import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from base import TextAPI


class InstructionAPI(TextAPI):
    """
    A class representing a Hugging Face API for generating text using a pre-trained instruction-tuned model.

    Attributes:
        model (Any): The pre-trained instruction-tuned model.
        tokenizer (Any): The tokenizer used to preprocess input text.
        model_name (str): The name of the pre-trained instruction-tuned model.
        model_revision (Optional[str]): The revision of the pre-trained instruction-tuned model.
        tokenizer_name (str): The name of the tokenizer used to preprocess input text.
        tokenizer_revision (Optional[str]): The revision of the tokenizer used to preprocess input text.
        model_class (str): The name of the class of the pre-trained instruction-tuned model.
        tokenizer_class (str): The name of the class of the tokenizer used to preprocess input text.
        use_cuda (bool): Whether to use a GPU for inference.
        quantization (int): The level of quantization to use for the pre-trained instruction-tuned model.
        precision (str): The precision to use for the pre-trained instruction-tuned model.
        device_map (str | Dict | None): The mapping of devices to use for inference.
        max_memory (Dict[int, str]): The maximum memory to use for inference.
        torchscript (bool): Whether to use a TorchScript-optimized version of the pre-trained instruction-tuned model.
        model_args (Any): Additional arguments to pass to the pre-trained instruction-tuned model.

    Methods:
        text(**kwargs: Any) -> Dict[str, Any]:
            Generates text based on the given prompt and decoding strategy.

        listen(model_name: str, model_class: str = "AutoModelForCausalLM", tokenizer_class: str = "AutoTokenizer", use_cuda: bool = False, precision: str = "float16", quantization: int = 0, device_map: str | Dict | None = "auto", max_memory={0: "24GB"}, torchscript: bool = True, endpoint: str = "*", port: int = 3000, cors_domain: str = "http://localhost:3000", username: Optional[str] = None, password: Optional[str] = None, **model_args: Any) -> None:
            Starts a CherryPy server to listen for requests to generate text.
    """

    model: Any
    tokenizer: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ):
        """
        Initializes a new instance of the TextAPI class.

        Args:
            input (BatchInput): The input data to process.
            output (BatchOutput): The output data to process.
            state (State): The state of the API.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def complete(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Completes text based on the given prompt and decoding strategy.

        Args:
            **kwargs (Any): Additional arguments to pass to the pre-trained instruction-tuned model.

        Returns:
            Dict[str, Any]: A dictionary containing the prompt, arguments, and generated text.
        """
        data = cherrypy.request.json
        prompt = data.get("prompt")
        decoding_strategy = data.get("decoding_strategy", "generate")

        generation_params = data
        if "decoding_strategy" in generation_params:
            del generation_params["decoding_strategy"]
        if "prompt" in generation_params:
            del generation_params["prompt"]

        return {
            "prompt": prompt,
            "args": data,
            "completion": self.generate(prompt=prompt, decoding_strategy=decoding_strategy, **generation_params),
        }
