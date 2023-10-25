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

from typing import Any, Dict, List, Optional, Tuple, Union

import cherrypy
import torch
import transformers
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)


class HuggingFaceAPI(Bolt):
    model: Any
    tokenizer: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
    ):
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def text(self, **kwargs: Any) -> Dict[str, Any]:
        data = cherrypy.request.json
        prompt = data.get("prompt")
        decoding_strategy = data.get("decoding_strategy", "generate")

        max_new_tokens = data.get("max_new_tokens")
        max_length = data.get("max_length")
        temperature = data.get("temperature")
        diversity_penalty = data.get("diversity_penalty")
        num_beams = data.get("num_beams")
        length_penalty = data.get("length_penalty")
        early_stopping = data.get("early_stopping")

        others = data.__dict__

        return {
            "prompt": prompt,
            "args": others,
            "completion": self.generate(
                prompt=prompt,
                decoding_strategy=decoding_strategy,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                temperature=temperature,
                diversity_penalty=diversity_penalty,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                **others,
            ),
        }

    def generate(
        self,
        prompt: str,
        decoding_strategy: str = "generate",
        **generation_params: Any,
    ) -> dict:
        """ """
        results: Dict[int, Dict[str, Union[str, List[str]]]] = {}
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id
        if not pad_token_id:
            pad_token_id = eos_token_id

        # Default parameters for each strategy
        default_params = {
            "generate": {"max_length": 4096},
            "greedy_search": {"max_length": 4096, "eos_token_id": eos_token_id, "pad_token_id": pad_token_id},
            "contrastive_search": {"max_length": 4096},
            "sample": {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "max_length": 4096},
            "beam_search": {"num_beams": 4, "max_length": 4096},
            "beam_sample": {"num_beams": 4, "temperature": 0.6, "max_length": 4096},
            "group_beam_search": {"num_beams": 4, "diversity_penalty": 0.5, "max_length": 4096},
            "constrained_beam_search": {"num_beams": 4, "max_length": 4096, "constraints": None},
        }

        # Merge default params with user-provided params
        strategy_params = {**default_params.get(decoding_strategy, {}), **generation_params}  # type: ignore

        # Prepare LogitsProcessorList and BeamSearchScorer for beam search strategies
        if decoding_strategy in ["beam_search", "beam_sample", "group_beam_search"]:
            logits_processor = LogitsProcessorList(
                [MinLengthLogitsProcessor(min_length=strategy_params.get("min_length", 0), eos_token_id=eos_token_id)]
            )
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                max_length=strategy_params.get("max_length", 20),
                num_beams=strategy_params.get("num_beams", 1),
                device=self.model.device,
                length_penalty=strategy_params.get("length_penalty", 1.0),
                do_early_stopping=strategy_params.get("early_stopping", False),
            )
            strategy_params.update({"logits_processor": logits_processor, "beam_scorer": beam_scorer})

            if decoding_strategy == "beam_sample":
                strategy_params.update({"logits_warper": LogitsProcessorList()})

        # Map of decoding strategy to method
        strategy_to_method = {
            "generate": self.model.generate,
            "greedy_search": self.model.greedy_search,
            "contrastive_search": self.model.contrastive_search,
            "sample": self.model.sample,
            "beam_search": self.model.beam_search,
            "beam_sample": self.model.beam_sample,
            "group_beam_search": self.model.group_beam_search,
            "constrained_beam_search": self.model.constrained_beam_search,
        }

        try:
            self.log.debug(f"Generating completion for prompt {prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            input_ids = input_ids.to(self.model.device)

            # Replicate input_ids for beam search
            if decoding_strategy in ["beam_search", "beam_sample", "group_beam_search"]:
                num_beams = strategy_params.get("num_beams", 1)
                input_ids = input_ids.repeat(num_beams, 1)

            # Use the specified decoding strategy
            decoding_method = strategy_to_method.get(decoding_strategy, self.model.generate)
            generated_ids = decoding_method(input_ids, **strategy_params)

            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            self.log.debug(f"Generated text: {generated_text}")

            return generated_text

        except Exception as e:
            self.log.exception(f"An error occurred: {e}")
            raise

    def load_models(
        self,
        model_name: str,
        tokenizer_name: str,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        model_class_name: str = "AutoModelForCausalLM",
        tokenizer_class_name: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
        **model_args: Any,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads a Hugging Face model and tokenizer optimized for inference.

        Parameters:
        - model_name (str): The name of the model to load.
        - model_class_name (str): The class name of the model to load. Default is "AutoModelForCausalLM".
        - tokenizer_class_name (str): The class name of the tokenizer to load. Default is "AutoTokenizer".
        - use_cuda (bool): Whether to use CUDA for GPU acceleration. Default is False.
        - precision (str): The bit precision for model and tokenizer. Options are 'float32', 'float16', 'bfloat16'. Default is 'float16'.
        - device_map (Union[str, Dict]): Device map for model placement. Default is "auto".
        - max_memory (Dict): Maximum GPU memory to be allocated.
        - model_args (Any): Additional keyword arguments for the model.

        Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

        Usage:
        ```python
        model, tokenizer = load_models("gpt-2", use_cuda=True, precision='float32', quantize=True, quantize_bits=8)
        ```
        """
        self.log.info(f"Loading Hugging Face model: {model_name}")

        # Determine the torch dtype based on precision
        if precision == "float16":
            torch_dtype = torch.float16
        elif precision == "float32":
            torch_dtype = torch.float32
        elif precision == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError("Unsupported precision. Choose from 'float32', 'float16', 'bfloat16'.")

        if use_cuda and not device_map:
            device_map = "auto"

        ModelClass = getattr(transformers, model_class_name)
        TokenizerClass = getattr(transformers, tokenizer_class_name)

        # Load the model and tokenizer
        tokenizer = TokenizerClass.from_pretrained(tokenizer_name, revision=tokenizer_revision, torch_dtype=torch_dtype)

        self.log.info(f"Loading model from {model_name} {model_revision} with {model_args}")
        if quantization == 8:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torchscript=torchscript,
                max_memory=max_memory,
                device_map=device_map,
                load_in_8bit=True,
                **model_args,
            )
        elif quantization == 4:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torchscript=torchscript,
                max_memory=max_memory,
                device_map=device_map,
                load_in_4bit=True,
                **model_args,
            )
        else:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch_dtype,
                torchscript=torchscript,
                max_memory=max_memory,
                device_map=device_map,
                **model_args,
            )

        # Set to evaluation mode for inference
        model.eval()

        if tokenizer and tokenizer.eos_token and (not tokenizer.pad_token):
            tokenizer.pad_token = tokenizer.eos_token

        self.log.debug("Hugging Face model and tokenizer loaded successfully.")
        return model, tokenizer

    def listen(
        self,
        model_name: str,
        model_class_name: str = "AutoModelForCausalLM",
        tokenizer_class_name: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
        endpoint: str = "*",
        port: int = 3000,
        cors_domain: str = "http://localhost:3000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **model_args: Any,
    ) -> None:
        self.model_name = model_name
        self.model_class_name = model_class_name
        self.tokenizer_class_name = tokenizer_class_name
        self.use_cuda = use_cuda
        self.quantization = quantization
        self.precision = precision
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.model_args = model_args

        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            tokenizer_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            tokenizer_name = model_name
        else:
            model_revision = None
            tokenizer_revision = None
        self.model_name = model_name
        self.model_revision = model_revision
        self.tokenizer_name = tokenizer_name
        self.tokenizer_revision = tokenizer_revision

        self.model, self.tokenizer = self.load_models(
            model_name=self.model_name,
            tokenizer_name=self.tokenizer_name,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            model_class_name=self.model_class_name,
            tokenizer_class_name=self.tokenizer_class_name,
            use_cuda=self.use_cuda,
            precision=self.precision,
            quantization=self.quantization,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            **self.model_args,
        )

        def CORS():
            cherrypy.response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            cherrypy.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            cherrypy.response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            cherrypy.response.headers["Access-Control-Allow-Credentials"] = "true"

            if cherrypy.request.method == "OPTIONS":
                cherrypy.response.status = 200
                return True

        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,
                "tools.CORS.on": True,
            }
        )

        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", {"/": {"tools.CORS.on": True}})
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()
