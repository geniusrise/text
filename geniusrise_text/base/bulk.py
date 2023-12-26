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


class TextBulk(Bolt):
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
        **kwargs,
    ):
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def generate(
        self,
        prompt: str,
        decoding_strategy: str = "generate",
        **generation_params: Any,
    ) -> str:
        """
        Generate text completion for the given prompt using the specified decoding strategy.

        Args:
            prompt (str): The prompt to generate text completion for.
            decoding_strategy (str, optional): The decoding strategy to use. Defaults to "generate".
            **generation_params (Any): Additional parameters to pass to the decoding strategy.

        Returns:
            str: The generated text completion.

        Raises:
            Exception: If an error occurs during generation.

        Supported decoding strategies:
            - "generate": Generate text using the model's default generation method.
            - "greedy_search": Generate text using greedy search decoding strategy.
            - "contrastive_search": Generate text using contrastive search decoding strategy.
            - "sample": Generate text using sampling decoding strategy.
            - "beam_search": Generate text using beam search decoding strategy.
            - "beam_sample": Generate text using beam search with sampling decoding strategy.
            - "group_beam_search": Generate text using group beam search decoding strategy.
            - "constrained_beam_search": Generate text using constrained beam search decoding strategy.

        Additional parameters for each decoding strategy:
            - "generate": {"max_length": int}
            - "greedy_search": {"max_length": int, "eos_token_id": int, "pad_token_id": int}
            - "contrastive_search": {"max_length": int}
            - "sample": {"do_sample": bool, "temperature": float, "top_p": float, "max_length": int}
            - "beam_search": {"num_beams": int, "max_length": int}
            - "beam_sample": {"num_beams": int, "temperature": float, "max_length": int}
            - "group_beam_search": {"num_beams": int, "diversity_penalty": float, "max_length": int}
            - "constrained_beam_search": {"num_beams": int, "max_length": int, "constraints": None}

        Note:
            - The `max_length` parameter specifies the maximum length of the generated text.
            - The `eos_token_id` parameter specifies the end-of-sequence token ID.
            - The `pad_token_id` parameter specifies the padding token ID.
            - The `do_sample` parameter specifies whether to use sampling during decoding.
            - The `temperature` parameter controls the randomness of the sampling.
            - The `top_p` parameter controls the diversity of the sampling.
            - The `num_beams` parameter specifies the number of beams to use during beam search.
            - The `diversity_penalty` parameter controls the diversity of the generated text during group beam search.
            - The `constraints` parameter specifies any constraints to apply during decoding.
        """
        results: Dict[int, Dict[str, Union[str, List[str]]]] = {}
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id
        if not pad_token_id:
            pad_token_id = eos_token_id
            self.model.config.pad_token_id = pad_token_id

        # Default parameters for each strategy
        default_params = {
            "generate": {
                "max_length": 20,  # Maximum length the generated tokens can have
                "max_new_tokens": None,  # Maximum number of tokens to generate, ignoring prompt tokens
                "min_length": 0,  # Minimum length of the sequence to be generated
                "min_new_tokens": None,  # Minimum number of tokens to generate, ignoring prompt tokens
                "early_stopping": False,  # Stopping condition for beam-based methods
                "max_time": None,  # Maximum time allowed for computation in seconds
                "do_sample": False,  # Whether to use sampling for generation
                "num_beams": 1,  # Number of beams for beam search
                "num_beam_groups": 1,  # Number of groups for beam search to ensure diversity
                "penalty_alpha": None,  # Balances model confidence and degeneration penalty in contrastive search
                "use_cache": True,  # Whether the model should use past key/values attentions to speed up decoding
                "temperature": 1.0,  # Modulates next token probabilities
                "top_k": 50,  # Number of highest probability tokens to keep for top-k-filtering
                "top_p": 1.0,  # Smallest set of most probable tokens with cumulative probability >= top_p
                "typical_p": 1.0,  # Conditional probability of predicting a target token next
                "epsilon_cutoff": 0.0,  # Tokens with a conditional probability > epsilon_cutoff will be sampled
                "eta_cutoff": 0.0,  # Eta sampling, a hybrid of locally typical sampling and epsilon sampling
                "diversity_penalty": 0.0,  # Penalty subtracted from a beam's score if it generates a token same as any other group
                "repetition_penalty": 1.0,  # Penalty for repetition of ngrams
                "encoder_repetition_penalty": 1.0,  # Penalty on sequences not in the original input
                "length_penalty": 1.0,  # Exponential penalty to the length for beam-based generation
                "no_repeat_ngram_size": 0,  # All ngrams of this size can only occur once
                "bad_words_ids": None,  # List of token ids that are not allowed to be generated
                "force_words_ids": None,  # List of token ids that must be generated
                "renormalize_logits": False,  # Renormalize the logits after applying all logits processors
                "constraints": None,  # Custom constraints for generation
                "forced_bos_token_id": None,  # Token ID to force as the first generated token
                "forced_eos_token_id": None,  # Token ID to force as the last generated token
                "remove_invalid_values": False,  # Remove possible NaN and inf outputs
                "exponential_decay_length_penalty": None,  # Exponentially increasing length penalty after a certain number of tokens
                "suppress_tokens": None,  # Tokens that will be suppressed during generation
                "begin_suppress_tokens": None,  # Tokens that will be suppressed at the beginning of generation
                "forced_decoder_ids": None,  # Mapping from generation indices to token indices that will be forced
                "sequence_bias": None,  # Maps a sequence of tokens to its bias term
                "guidance_scale": None,  # Guidance scale for classifier free guidance (CFG)
                "low_memory": None,  # Switch to sequential topk for contrastive search to reduce peak memory
                "num_return_sequences": 1,  # Number of independently computed returned sequences for each batch element
                "output_attentions": False,  # Whether to return the attentions tensors of all layers
                "output_hidden_states": False,  # Whether to return the hidden states of all layers
                "output_scores": False,  # Whether to return the prediction scores
                "return_dict_in_generate": False,  # Whether to return a ModelOutput instead of a plain tuple
                "pad_token_id": None,  # The id of the padding token
                "bos_token_id": None,  # The id of the beginning-of-sequence token
                "eos_token_id": None,  # The id of the end-of-sequence token
            },
            "greedy_search": {"max_length": 4096, "eos_token_id": eos_token_id, "pad_token_id": pad_token_id},
            "contrastive_search": {"max_length": 4096},
            "sample": {"do_sample": True, "temperature": 0.6, "top_k": 50, "top_p": 0.9, "max_length": 4096},
            "beam_search": {"num_beams": 4, "max_length": 4096},
            "beam_sample": {"num_beams": 4, "temperature": 0.6, "max_length": 4096},
            "group_beam_search": {"num_beams": 4, "diversity_penalty": 0.5, "max_length": 4096},
            "constrained_beam_search": {"num_beams": 4, "max_length": 4096, "constraints": None},
        }

        # Merge default params with user-provided params
        strategy_params = {**default_params.get(decoding_strategy, {})}
        for k, v in generation_params.items():
            if k in strategy_params:
                strategy_params[k] = v

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
        awq_enabled: bool = False,
        **model_args: Any,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads a Hugging Face model and tokenizer optimized for inference.

        Args:
        - model_name (str): The name of the model to load.
        - tokenizer_name (str): The name of the tokenizer to load.
        - model_revision (Optional[str]): The revision of the model to load. Default is None.
        - tokenizer_revision (Optional[str]): The revision of the tokenizer to load. Default is None.
        - model_class (str): The class name of the model to load. Default is "AutoModelForCausalLM".
        - tokenizer_class (str): The class name of the tokenizer to load. Default is "AutoTokenizer".
        - use_cuda (bool): Whether to use CUDA for GPU acceleration. Default is False.
        - precision (str): The bit precision for model and tokenizer. Options are 'float32', 'float16', 'bfloat16'. Default is 'float16'.
        - quantization (int): The number of bits to use for quantization. Default is 0.
        - device_map (Union[str, Dict, None]): Device map for model placement. Default is "auto".
        - max_memory (Dict): Maximum GPU memory to be allocated. Default is {0: "24GB"}.
        - torchscript (bool): Whether to use TorchScript for model optimization. Default is True.
        - awq_enabled (bool): Whether to use AWQ for model optimization. Default is False.
        - model_args (Any): Additional keyword arguments for the model.

        Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

        Usage:
        ```python
        model, tokenizer = load_models("gpt-2", "gpt-2", use_cuda=True, precision='float32', quantization=8)
        ```
        """
        self.log.info(f"Loading Hugging Face model: {model_name}")

        # Determine the torch dtype based on precision
        if precision == "float32":
            torch_dtype = torch.float32
        elif precision == "float":
            torch_dtype = torch.float
        elif precision == "float64":
            torch_dtype = torch.float64
        elif precision == "double":
            torch_dtype = torch.double
        elif precision == "float16":
            torch_dtype = torch.float16
        elif precision == "bfloat16":
            torch_dtype = torch.bfloat16
        elif precision == "half":
            torch_dtype = torch.half
        elif precision == "uint8":
            torch_dtype = torch.uint8
        elif precision == "int8":
            torch_dtype = torch.int8
        elif precision == "int16":
            torch_dtype = torch.int16
        elif precision == "short":
            torch_dtype = torch.short
        elif precision == "int32":
            torch_dtype = torch.int32
        elif precision == "int":
            torch_dtype = torch.int
        elif precision == "int64":
            torch_dtype = torch.int64
        elif precision == "quint8":
            torch_dtype = torch.quint8
        elif precision == "qint8":
            torch_dtype = torch.qint8
        elif precision == "qint32":
            torch_dtype = torch.qint32
        else:
            torch_dtype = None
            raise ValueError("Unsupported precision. Choose from 'float32', 'float16', 'bfloat16'.")

        if use_cuda and not device_map:
            device_map = "auto"

        if awq_enabled:
            ModelClass = AutoModelForCausalLM
            self.log.info("AWQ Enabled: Loading AWQ Model")
        else:
            ModelClass = getattr(transformers, model_class)
        TokenizerClass = getattr(transformers, tokenizer_class)

        # Load the model and tokenizer
        tokenizer = TokenizerClass.from_pretrained(tokenizer_name, revision=tokenizer_revision, torch_dtype=torch_dtype)

        self.log.info(f"Loading model from {model_name} {model_revision} with {model_args}")
        if awq_enabled and quantization > 0:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch_dtype,
                **model_args,
            )
        elif quantization == 8:
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

        eos_token_id = model.config.eos_token_id
        pad_token_id = model.config.pad_token_id
        if not pad_token_id:
            model.config.pad_token_id = eos_token_id

        self.log.debug("Hugging Face model and tokenizer loaded successfully.")
        return model, tokenizer
