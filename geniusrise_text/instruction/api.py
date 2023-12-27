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
    r"""
    InstructionAPI is designed for generating text based on prompts using instruction-tuned language models.
    It serves as an interface to Hugging Face's pre-trained instruction-tuned models, providing a flexible API
    for various text generation tasks. It can be used in scenarios ranging from generating creative content to
    providing instructions or answers based on the prompts.

    Attributes:
        model (Any): The loaded instruction-tuned language model.
        tokenizer (Any): The tokenizer for processing text suitable for the model.

    Methods:
        complete(**kwargs: Any) -> Dict[str, Any]:
            Generates text based on the given prompt and decoding strategy.

        listen(**model_args: Any) -> None:
            Starts a server to listen for text generation requests.

    CLI Usage Example:
    ```bash
    genius InstructionAPI rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder none \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder none \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise\
            --postgres_table state \
        listen \
            --args \
                model_name="TheBloke/Mistral-7B-OpenOrca-AWQ" \
                model_class="AutoModelForCausalLM" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="float16" \
                quantization=0 \
                device_map="auto" \
                max_memory=None \
                torchscript=False \
                awq_enabled=True \
                flash_attention=True \
                endpoint="*" \
                port=3001 \
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
        Initializes a new instance of the InstructionAPI class, setting up the necessary configurations
        for input, output, and state.

        Args:
            input (BatchInput): Configuration for the input data.
            output (BatchOutput): Configuration for the output data.
            state (State): The state of the API.
            **kwargs (Any): Additional keyword arguments for extended functionality.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def complete(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Handles POST requests to generate text based on the given prompt and decoding strategy. It uses the pre-trained
        model specified in the setup to generate a completion for the input prompt.

        Args:
            **kwargs (Any): Arbitrary keyword arguments containing the 'prompt' and other parameters for text generation.

        Returns:
            Dict[str, Any]: A dictionary containing the original prompt and the generated completion.

        Example CURL Requests:
        ```bash
        /usr/bin/curl -X POST localhost:3001/api/v1/complete \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "<|system|>\n<|end|>\n<|user|>\nHow do I sort a list in Python?<|end|>\n<|assistant|>",
                "decoding_strategy": "generate",
                "max_new_tokens": 100,
                "do_sample": true,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95
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
