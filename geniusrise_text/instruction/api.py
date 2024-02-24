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

import asyncio
import cherrypy
from concurrent.futures import ThreadPoolExecutor
from geniusrise_text.base import TextAPI
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


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
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
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

    Or using VLLM:
    ```bash
    genius InstructionAPI rise \
        batch \
                --input_folder ./input \
        batch \
                --output_folder ./output \
        none \
        --id mistralai/Mistral-7B-Instruct-v0.1 \
        listen \
            --args \
                model_name="mistralai/Mistral-7B-Instruct-v0.1" \
                model_class="AutoModelForCausalLM" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="bfloat16" \
                quantization=0 \
                device_map="auto" \
                max_memory=None \
                torchscript=False \
                use_vllm=True \
                vllm_enforce_eager=True \
                vllm_max_model_len=1024 \
                concurrent_queries=False \
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
        self.hf_pipeline = None
        self.vllm_server: Optional[OpenAIServingChat] = None
        self.event_loop: Any = None
        self.executor = ThreadPoolExecutor(max_workers=4)

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

    def initialize_pipeline(self):
        """
        Lazy initialization of the Hugging Face pipeline for chat interaction.
        """
        if not self.hf_pipeline:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.use_cuda:
                model.cuda()
            self.hf_pipeline = pipeline("conversational", model=model, tokenizer=tokenizer)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def chat(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Handles chat interaction using the Hugging Face pipeline. This method enables conversational text generation,
        simulating a chat-like interaction based on user and system prompts.

        Args:
            **kwargs (Any): Arbitrary keyword arguments containing 'user_prompt' and 'system_prompt'.

        Returns:
            Dict[str, Any]: A dictionary containing the user prompt, system prompt, and chat interaction results.

        Example CURL Request for chat interaction:
        ```bash
        /usr/bin/curl -X POST localhost:3001/api/v1/chat \
            -H "Content-Type: application/json" \
            -d '{
                "user_prompt": "What is the capital of France?",
                "system_prompt": "The capital of France is"
            }' | jq
        ```
        """
        self.initialize_pipeline()  # Initialize the pipeline on first API hit

        data = cherrypy.request.json
        user_prompt = data.get("user_prompt")
        system_prompt = data.get("system_prompt")

        result = self.hf_pipeline(user_prompt, system_prompt)  # type: ignore

        return {"user_prompt": user_prompt, "system_prompt": system_prompt, "result": result}

    def initialize_vllm(self, chat_template: str, response_role: str = "assistant"):
        self.vllm_server = OpenAIServingChat(
            engine=self.model, served_model=self.model_name, response_role=response_role, chat_template=chat_template
        )
        self.event_loop = asyncio.new_event_loop()

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def chat_vllm(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Handles POST requests to generate chat completions using the VLLM (Versatile Language Learning Model) engine.
        This method accepts various parameters for customizing the chat completion request, including message content,
        generation settings, and more.

        Parameters:
        - **kwargs (Any): Arbitrary keyword arguments. Expects data in JSON format containing any of the following keys:
            - messages (Union[str, List[Dict[str, str]]]): The messages for the chat context.
            - temperature (float, optional): The sampling temperature. Defaults to 0.7.
            - top_p (float, optional): The nucleus sampling probability. Defaults to 1.0.
            - n (int, optional): The number of completions to generate. Defaults to 1.
            - max_tokens (int, optional): The maximum number of tokens to generate.
            - stop (Union[str, List[str]], optional): Stop sequence to end generation.
            - stream (bool, optional): Whether to stream the response. Defaults to False.
            - presence_penalty (float, optional): The presence penalty. Defaults to 0.0.
            - frequency_penalty (float, optional): The frequency penalty. Defaults to 0.0.
            - logit_bias (Dict[str, float], optional): Adjustments to the logits of specified tokens.
            - user (str, optional): An identifier for the user making the request.
            - (Additional model-specific parameters)

        Returns:
        Dict[str, Any]: A dictionary with the chat completion response or an error message.

        Example CURL Request:
        ```bash
        curl -X POST "http://localhost:3000/chat_vllm" \
            -H "Content-Type: application/json" \
            -d '{
                "messages": [
                    {"role": "user", "content": "Whats the weather like in London?"}
                ],
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "max_tokens": 50,
                "stream": false,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "logit_bias": {},
                "user": "example_user"
            }'
        ```
        This request asks the VLLM engine to generate a completion for the provided chat context, with specified generation settings.
        """
        # Extract data from the POST request
        data = cherrypy.request.json
        response_role = data.get("response_role", "assistant")
        chat_template = data.get("chat_template", None)

        # Initialize VLLM server with chat template and response role if not already initialized
        if not hasattr(self, "vllm_server") or self.vllm_server is None:
            self.initialize_vllm(chat_template=chat_template, response_role=response_role)

        # Prepare the chat completion request
        chat_request = ChatCompletionRequest(
            model=self.model_name,
            messages=data.get("messages"),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 1.0),
            n=data.get("n", 1),
            max_tokens=data.get("max_tokens"),
            stop=data.get("stop", []),
            stream=data.get("stream", False),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            logit_bias=data.get("logit_bias", {}),
            user=data.get("user"),
            best_of=data.get("best_of"),
            top_k=data.get("top_k", -1),
            ignore_eos=data.get("ignore_eos", False),
            use_beam_search=data.get("use_beam_search", False),
            stop_token_ids=data.get("stop_token_ids", []),
            skip_special_tokens=data.get("skip_special_tokens", True),
            spaces_between_special_tokens=data.get("spaces_between_special_tokens", True),
            add_generation_prompt=data.get("add_generation_prompt", True),
            echo=data.get("echo", False),
            repetition_penalty=data.get("repetition_penalty", 1.0),
            min_p=data.get("min_p", 0.0),
            include_stop_str_in_output=data.get("include_stop_str_in_output", False),
            length_penalty=data.get("length_penalty", 1.0),
        )

        # Generate chat completion using the VLLM engine
        try:

            class DummyObject:
                async def is_disconnected(self):
                    return False

            async def async_call():
                response = await self.vllm_server.create_chat_completion(
                    request=chat_request, raw_request=DummyObject()
                )
                return response

            chat_completion = asyncio.run(async_call())

            return chat_completion.model_dump() if chat_completion else {"error": "Failed to generate chat completion"}
        except Exception as e:
            self.log.exception("Error generating chat completion: %s", str(e))
            raise e
