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


class HuggingFaceBulk(Bolt):
    """
    A class that provides bulk text generation functionality using Hugging Face models.

    Attributes:
        model (Any): The Hugging Face model to use for text generation.
        tokenizer (Any): The Hugging Face tokenizer to use for text generation.

    Args:
        input (BatchInput): The input data to process.
        output (BatchOutput): The output data to return.
        state (State): The state of the Bolt.

    Methods:
        text(**kwargs: Any) -> Dict[str, Any]:
            Exposes the text generation functionality as a REST API endpoint.
            Accepts a JSON payload with the following keys:
                - prompt (str): The prompt to generate text from.
                - decoding_strategy (str): The decoding strategy to use for text generation.
                - max_new_tokens (int): The maximum number of new tokens to generate.
                - max_length (int): The maximum length of the generated text.
                - temperature (float): The temperature to use for sampling-based decoding strategies.
                - diversity_penalty (float): The diversity penalty to use for beam search-based decoding strategies.
                - num_beams (int): The number of beams to use for beam search-based decoding strategies.
                - length_penalty (float): The length penalty to use for beam search-based decoding strategies.
                - early_stopping (bool): Whether to stop decoding early for beam search-based decoding strategies.
                - Any other key-value pairs will be passed as generation parameters to the Hugging Face model.
            Returns a JSON payload with the following keys:
                - prompt (str): The prompt used for text generation.
                - args (Dict[str, Any]): The generation parameters used for text generation.
                - completion (str): The generated text.

        generate(prompt: str, decoding_strategy: str = "generate", **generation_params: Any) -> dict:
            Generates text using the specified decoding strategy and generation parameters.
            Returns a dictionary with the following keys:
                - prompt (str): The prompt used for text generation.
                - completion (str): The generated text.
    """

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
        model_class: str = "AutoModelForCausalLM",
        tokenizer_class: str = "AutoTokenizer",
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
        - model_class (str): The class name of the model to load. Default is "AutoModelForCausalLM".
        - tokenizer_class (str): The class name of the tokenizer to load. Default is "AutoTokenizer".
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

        ModelClass = getattr(transformers, model_class)
        TokenizerClass = getattr(transformers, tokenizer_class)

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
