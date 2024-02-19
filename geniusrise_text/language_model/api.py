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

from geniusrise_text.base import TextAPI


class LanguageModelAPI(TextAPI):
    r"""
    LanguageModelAPI is a class for interacting with pre-trained language models to generate text. It allows for
    customizable text generation via a CherryPy web server, handling requests and generating responses using
    a specified language model. This class is part of the GeniusRise ecosystem for facilitating NLP tasks.

    Attributes:
        model (Any): The loaded language model used for text generation.
        tokenizer (Any): The tokenizer corresponding to the language model, used for processing input text.

    Methods:
        complete(**kwargs: Any) -> Dict[str, Any]: Generates text based on provided prompts and model parameters.

    CLI Usage Example:
    ```bash
    genius LanguageModelAPI rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        --id mistralai/Mistral-7B-v0.1-lol \
        listen \
            --args \
                model_name="mistralai/Mistral-7B-v0.1" \
                model_class="AutoModelForCausalLM" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="float16" \
                quantization=0 \
                device_map="auto" \
                max_memory=None \
                torchscript=False \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```
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
        Initializes the LanguageModelAPI with configurations for the input, output, and state management,
        along with any additional model-specific parameters.

        Args:
            input (BatchInput): The configuration for input data handling.
            output (BatchOutput): The configuration for output data handling.
            state (State): The state management for the API.
            **kwargs (Any): Additional keyword arguments for model configuration and API setup.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def complete(self, **kwargs: Any) -> Dict[str, Any]:
        r"""
        Handles POST requests to generate text based on a given prompt and model-specific parameters. This method
        is exposed as a web endpoint through CherryPy and returns a JSON response containing the original prompt,
        the generated text, and any additional returned information from the model.

        Args:
            **kwargs (Any): Arbitrary keyword arguments containing the prompt, and any additional parameters
            for the text generation model.

        Returns:
            Dict[str, Any]: A dictionary with the original prompt, generated text, and other model-specific information.

        Example CURL Request:
        ```bash
        /usr/bin/curl -X POST localhost:3000/api/v1/complete \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a PRD for Oauth auth using keycloak\n\n### Response:",
                "decoding_strategy": "generate",
                "max_new_tokens": 1024,
                "do_sample": true
            }' | jq
        ```
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
