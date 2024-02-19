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
import os
import torch
import transformers
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from optimum.bettertransformer import BetterTransformer

from geniusrise_text.base.communication import send_email


class TextBulk(Bolt):
    """
    TextBulk is a foundational class for enabling bulk processing of text with various generation models.
    It primarily focuses on using Hugging Face models to provide a robust and efficient framework for
    large-scale text generation tasks. The class supports various decoding strategies to generate text
    that can be tailored to specific needs or preferences.

    Attributes:
        model (AutoModelForCausalLM): The language model for text generation.
        tokenizer (AutoTokenizer): The tokenizer for preparing input data for the model.

    Args:
        input (BatchInput): Configuration and data inputs for the batch process.
        output (BatchOutput): Configurations for output data handling.
        state (State): State management for the Bolt.
        **kwargs: Arbitrary keyword arguments for extended configurations.

    Methods:
        text(**kwargs: Any) -> Dict[str, Any]:
            Provides an API endpoint for text generation functionality.
            Accepts various parameters for customizing the text generation process.

        generate(prompt: str, decoding_strategy: str = "generate", **generation_params: Any) -> dict:
            Generates text based on the provided prompt and parameters. Supports multiple decoding strategies for diverse applications.

    The class serves as a versatile tool for text generation, supporting various models and configurations.
    It can be extended or used as is for efficient text generation tasks.
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
        """
        Initializes the TextBulk with configurations and sets up logging. It prepares the environment for text generation tasks.

        Args:
            input (BatchInput): The input data configuration for the text generation task.
            output (BatchOutput): The output data configuration for the results of the text generation.
            state (State): The state configuration for the Bolt, managing its operational status.
            **kwargs: Additional keyword arguments for extended functionality and model configurations.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def generate(
        self,
        prompt: str,
        decoding_strategy: str = "generate",
        **generation_params: Any,
    ) -> str:
        r"""
        Generate text completion for the given prompt using the specified decoding strategy.

        Args:
            prompt (str): The prompt to generate text completion for.
            decoding_strategy (str, optional): The decoding strategy to use. Defaults to "generate".
            **generation_params (Any): Additional parameters to pass to the decoding strategy.

        Returns:
            str: The generated text completion.

        Raises:
            Exception: If an error occurs during generation.

        Supported decoding strategies and their additional parameters:
            - "generate": Uses the model's default generation method. (Parameters: max_length, num_beams, etc.)
            - "greedy_search": Generates text using a greedy search decoding strategy.
            Parameters: max_length, eos_token_id, pad_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "contrastive_search": Generates text using contrastive search decoding strategy.
            Parameters: top_k, penalty_alpha, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus, sequential.
            - "sample": Generates text using a sampling decoding strategy.
            Parameters: do_sample, temperature, top_k, top_p, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "beam_search": Generates text using beam search decoding strategy.
            Parameters: num_beams, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "beam_sample": Generates text using beam search with sampling decoding strategy.
            Parameters: num_beams, temperature, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "group_beam_search": Generates text using group beam search decoding strategy.
            Parameters: num_beams, diversity_penalty, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.
            - "constrained_beam_search": Generates text using constrained beam search decoding strategy.
            Parameters: num_beams, max_length, constraints, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus.

        All generation parameters:
            - max_length: Maximum length the generated tokens can have
            - max_new_tokens: Maximum number of tokens to generate, ignoring prompt tokens
            - min_length: Minimum length of the sequence to be generated
            - min_new_tokens: Minimum number of tokens to generate, ignoring prompt tokens
            - early_stopping: Stopping condition for beam-based methods
            - max_time: Maximum time allowed for computation in seconds
            - do_sample: Whether to use sampling for generation
            - num_beams: Number of beams for beam search
            - num_beam_groups: Number of groups for beam search to ensure diversity
            - penalty_alpha: Balances model confidence and degeneration penalty in contrastive search
            - use_cache: Whether the model should use past key/values attentions to speed up decoding
            - temperature: Modulates next token probabilities
            - top_k: Number of highest probability tokens to keep for top-k-filtering
            - top_p: Smallest set of most probable tokens with cumulative probability >= top_p
            - typical_p: Conditional probability of predicting a target token next
            - epsilon_cutoff: Tokens with a conditional probability > epsilon_cutoff will be sampled
            - eta_cutoff: Eta sampling, a hybrid of locally typical sampling and epsilon sampling
            - diversity_penalty: Penalty subtracted from a beam's score if it generates a token same as any other group
            - repetition_penalty: Penalty for repetition of ngrams
            - encoder_repetition_penalty: Penalty on sequences not in the original input
            - length_penalty: Exponential penalty to the length for beam-based generation
            - no_repeat_ngram_size: All ngrams of this size can only occur once
            - bad_words_ids: List of token ids that are not allowed to be generated
            - force_words_ids: List of token ids that must be generated
            - renormalize_logits: Renormalize the logits after applying all logits processors
            - constraints: Custom constraints for generation
            - forced_bos_token_id: Token ID to force as the first generated token
            - forced_eos_token_id: Token ID to force as the last generated token
            - remove_invalid_values: Remove possible NaN and inf outputs
            - exponential_decay_length_penalty: Exponentially increasing length penalty after a certain number of tokens
            - suppress_tokens: Tokens that will be suppressed during generation
            - begin_suppress_tokens: Tokens that will be suppressed at the beginning of generation
            - forced_decoder_ids: Mapping from generation indices to token indices that will be forced
            - sequence_bias: Maps a sequence of tokens to its bias term
            - guidance_scale: Guidance scale for classifier free guidance (CFG)
            - low_memory: Switch to sequential topk for contrastive search to reduce peak memory
            - num_return_sequences: Number of independently computed returned sequences for each batch element
            - output_attentions: Whether to return the attentions tensors of all layers
            - output_hidden_states: Whether to return the hidden states of all layers
            - output_scores: Whether to return the prediction scores
            - return_dict_in_generate: Whether to return a ModelOutput instead of a plain tuple
            - pad_token_id: The id of the padding token
            - bos_token_id: The id of the beginning-of-sequence token
            - eos_token_id: The id of the end-of-sequence token
            - max_length: The maximum length of the sequence to be generated
            - eos_token_id: End-of-sequence token ID
            - pad_token_id: Padding token ID
            - output_attentions: Return attention tensors of all attention layers if True
            - output_hidden_states: Return hidden states of all layers if True
            - output_scores: Return prediction scores if True
            - return_dict_in_generate: Return a ModelOutput instead of a plain tuple if True
            - synced_gpus: Continue running the while loop until max_length for ZeRO stage 3 if True
            - top_k: Size of the candidate set for re-ranking in contrastive search
            - penalty_alpha: Degeneration penalty; active when larger than 0
            - eos_token_id: End-of-sequence token ID(s)
            - sequential: Switch to sequential topk hidden state computation to reduce memory if True
            - do_sample: Use sampling for generation if True
            - temperature: Temperature for sampling
            - top_p: Cumulative probability for top-p-filtering
            - diversity_penalty: Penalty for reducing similarity across different beam groups
            - constraints: List of constraints to apply during beam search
            - synced_gpus: Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
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
            "greedy_search": {
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "eos_token_id": eos_token_id,  # End-of-sequence token ID
                "pad_token_id": pad_token_id,  # Padding token ID
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "contrastive_search": {
                "top_k": 1,  # Size of the candidate set for re-ranking in contrastive search
                "penalty_alpha": 0,  # Degeneration penalty; active when larger than 0
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
                "sequential": False,  # Switch to sequential topk hidden state computation to reduce memory if True
            },
            "sample": {
                "do_sample": True,  # Use sampling for generation if True
                "temperature": 0.6,  # Temperature for sampling
                "top_k": 50,  # Number of highest probability tokens to keep for top-k-filtering
                "top_p": 0.9,  # Cumulative probability for top-p-filtering
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "beam_search": {
                "num_beams": 4,  # Number of beams for beam search
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "beam_sample": {
                "num_beams": 4,  # Number of beams for beam search
                "temperature": 0.6,  # Temperature for sampling
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "group_beam_search": {
                "num_beams": 4,  # Number of beams for beam search
                "diversity_penalty": 0.5,  # Penalty for reducing similarity across different beam groups
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Continue running the while loop until max_length for ZeRO stage 3 if True
            },
            "constrained_beam_search": {
                "num_beams": 4,  # Number of beams for beam search
                "max_length": 4096,  # The maximum length of the sequence to be generated
                "constraints": None,  # List of constraints to apply during beam search
                "pad_token_id": pad_token_id,  # Padding token ID
                "eos_token_id": eos_token_id,  # End-of-sequence token ID(s)
                "output_attentions": False,  # Return attention tensors of all attention layers if True
                "output_hidden_states": False,  # Return hidden states of all layers if True
                "output_scores": False,  # Return prediction scores if True
                "return_dict_in_generate": False,  # Return a ModelOutput instead of a plain tuple if True
                "synced_gpus": False,  # Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            },
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

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """
        Determines the torch dtype based on the specified precision.

        Args:
            precision (str): The desired precision for computations.

        Returns:
            torch.dtype: The corresponding torch dtype.

        Raises:
            ValueError: If an unsupported precision is specified.
        """
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float,
            "float64": torch.float64,
            "double": torch.double,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "half": torch.half,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "short": torch.short,
            "int32": torch.int32,
            "int": torch.int,
            "int64": torch.int64,
            "quint8": torch.quint8,
            "qint8": torch.qint8,
            "qint32": torch.qint32,
        }
        return dtype_map.get(precision, torch.float)

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
        torchscript: bool = False,
        compile: bool = False,
        awq_enabled: bool = False,
        flash_attention: bool = False,
        better_transformers: bool = False,
        **model_args: Any,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads and configures the specified model and tokenizer for text generation. It ensures the models are optimized for inference.

        Args:
            model_name (str): The name or path of the model to load.
            tokenizer_name (str): The name or path of the tokenizer to load.
            model_revision (Optional[str]): The specific model revision to load (e.g., a commit hash).
            tokenizer_revision (Optional[str]): The specific tokenizer revision to load (e.g., a commit hash).
            model_class (str): The class of the model to be loaded.
            tokenizer_class (str): The class of the tokenizer to be loaded.
            use_cuda (bool): Flag to utilize CUDA for GPU acceleration.
            precision (str): The desired precision for computations ("float32", "float16", etc.).
            quantization (int): The bit level for model quantization (0 for none, 8 for 8-bit quantization).
            device_map (str | Dict | None): The specific device(s) to use for model operations.
            max_memory (Dict): A dictionary defining the maximum memory to allocate for the model.
            torchscript (bool): Flag to enable TorchScript for model optimization.
            compile (bool): Flag to enable JIT compilation of the model.
            awq_enabled (bool): Flag to enable AWQ (Adaptive Weight Quantization).
            flash_attention (bool): Flag to enable Flash Attention optimization for faster processing.
            better_transformers (bool): Flag to enable Better Transformers optimization for faster processing.
            **model_args (Any): Additional arguments to pass to the model during its loading.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer ready for text generation.
        """
        self.log.info(f"Loading Hugging Face model: {model_name}")

        # Determine the torch dtype based on precision
        torch_dtype = self._get_torch_dtype(precision)

        if use_cuda and not device_map:
            device_map = "auto"

        if awq_enabled:
            ModelClass = AutoModelForCausalLM
            self.log.info("AWQ Enabled: Loading AWQ Model")
        else:
            ModelClass = getattr(transformers, model_class)
        TokenizerClass = getattr(transformers, tokenizer_class)

        # Load the model and tokenizer
        if model_name == "local":
            tokenizer = TokenizerClass.from_pretrained(
                os.path.join(self.input.get(), "/model"), torch_dtype=torch_dtype
            )
        else:
            tokenizer = TokenizerClass.from_pretrained(
                tokenizer_name, revision=tokenizer_revision, torch_dtype=torch_dtype
            )

        if flash_attention:
            model_args = {**model_args, **{"attn_implementation": "flash_attention_2"}}

        self.log.info(f"Loading model from {model_name} {model_revision} with {model_args}")
        if awq_enabled and quantization > 0:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    os.path.join(self.input.get(), "/model"),
                    torch_dtype=torch_dtype,
                    **model_args,
                )
            else:
                model = ModelClass.from_pretrained(
                    model_name,
                    revision=model_revision,
                    torch_dtype=torch_dtype,
                    **model_args,
                )
        elif quantization == 8:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    os.path.join(self.input.get(), "/model"),
                    torchscript=torchscript,
                    max_memory=max_memory,
                    device_map=device_map,
                    load_in_8bit=True,
                    **model_args,
                )
            else:
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
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    os.path.join(self.input.get(), "/model"),
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
                    torchscript=torchscript,
                    max_memory=max_memory,
                    device_map=device_map,
                    load_in_4bit=True,
                    **model_args,
                )
        else:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    os.path.join(self.input.get(), "/model"),
                    torch_dtype=torch_dtype,
                    torchscript=torchscript,
                    max_memory=max_memory,
                    device_map=device_map,
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

        if compile and not torchscript:
            model = torch.compile(model)

        if better_transformers:
            model = BetterTransformer.transform(model, keep_original_model=True)

        # Set to evaluation mode for inference
        model.eval()

        if tokenizer and tokenizer.eos_token and (not tokenizer.pad_token):
            tokenizer.pad_token = tokenizer.eos_token

        eos_token_id = model.config.eos_token_id
        pad_token_id = model.config.pad_token_id
        if not pad_token_id:
            model.config.pad_token_id = eos_token_id

        self.log.debug("Text model and tokenizer loaded successfully.")
        return model, tokenizer

    def done(self):
        if self.notification_email:
            self.output.flush()
            send_email(recipient=self.notification_email, bucket_name=self.output.bucket, prefix=self.output.s3_folder)
