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

import json
import threading
from typing import Any, Dict, List, Optional, Union

import cherrypy
import llama_cpp
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger

from .bulk import TextBulk

# Define a global lock for sequential access control
sequential_lock = threading.Lock()


class TextAPI(TextBulk):
    """
    A class representing a Hugging Face API for generating text using a pre-trained language model.

    Attributes:
        model (Any): The pre-trained language model.
        tokenizer (Any): The tokenizer used to preprocess input text.
        model_name (str): The name of the pre-trained language model.
        model_revision (Optional[str]): The revision of the pre-trained language model.
        tokenizer_name (str): The name of the tokenizer used to preprocess input text.
        tokenizer_revision (Optional[str]): The revision of the tokenizer used to preprocess input text.
        model_class (str): The name of the class of the pre-trained language model.
        tokenizer_class (str): The name of the class of the tokenizer used to preprocess input text.
        use_cuda (bool): Whether to use a GPU for inference.
        quantization (int): The level of quantization to use for the pre-trained language model.
        precision (str): The precision to use for the pre-trained language model.
        device_map (str | Dict | None): The mapping of devices to use for inference.
        max_memory (Dict[int, str]): The maximum memory to use for inference.
        torchscript (bool): Whether to use a TorchScript-optimized version of the pre-trained language model.
        model_args (Any): Additional arguments to pass to the pre-trained language model.

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
    def text(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generates text based on the given prompt and decoding strategy.

        Args:
            **kwargs (Any): Additional arguments to pass to the pre-trained language model.

        Returns:
            Dict[str, Any]: A dictionary containing the prompt, arguments, and generated text.
        """
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

    def validate_password(self, realm, username, password):
        """
        Validate the username and password against expected values.

        Args:
            realm (str): The authentication realm.
            username (str): The provided username.
            password (str): The provided password.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        return username == self.username and password == self.password

    def listen(
        self,
        model_name: str,
        # Huggingface params
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
        concurrent_queries: bool = False,
        use_vllm: bool = False,
        use_llama_cpp: bool = False,
        # VLLM params
        vllm_tokenizer_mode: str = "auto",
        vllm_download_dir: Optional[str] = None,
        vllm_load_format: str = "auto",
        vllm_seed: int = 42,
        vllm_max_model_len: int = 1024,
        vllm_enforce_eager: bool = False,
        vllm_max_context_len_to_capture: int = 8192,
        vllm_block_size: int = 16,
        vllm_gpu_memory_utilization: float = 0.90,
        vllm_swap_space: int = 4,
        vllm_sliding_window: Optional[int] = None,
        vllm_pipeline_parallel_size: int = 1,
        vllm_tensor_parallel_size: int = 1,
        vllm_worker_use_ray: bool = False,
        vllm_max_parallel_loading_workers: Optional[int] = None,
        vllm_disable_custom_all_reduce: bool = False,
        vllm_max_num_batched_tokens: Optional[int] = None,
        vllm_max_num_seqs: int = 64,
        vllm_max_paddings: int = 512,
        vllm_max_lora_rank: Optional[int] = None,
        vllm_max_loras: Optional[int] = None,
        vllm_max_cpu_loras: Optional[int] = None,
        vllm_lora_extra_vocab_size: int = 0,
        vllm_placement_group: Optional[dict] = None,
        vllm_log_stats: bool = False,
        # llama.cpp params
        llama_cpp_filename: Optional[str] = None,
        llama_cpp_n_gpu_layers: int = 0,
        llama_cpp_split_mode: int = llama_cpp.LLAMA_SPLIT_LAYER,
        llama_cpp_tensor_split: Optional[List[float]] = None,
        llama_cpp_vocab_only: bool = False,
        llama_cpp_use_mmap: bool = True,
        llama_cpp_use_mlock: bool = False,
        llama_cpp_kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None,
        llama_cpp_seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        llama_cpp_n_ctx: int = 2048,
        llama_cpp_n_batch: int = 512,
        llama_cpp_n_threads: Optional[int] = None,
        llama_cpp_n_threads_batch: Optional[int] = None,
        llama_cpp_rope_scaling_type: Optional[int] = llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED,
        llama_cpp_rope_freq_base: float = 0.0,
        llama_cpp_rope_freq_scale: float = 0.0,
        llama_cpp_yarn_ext_factor: float = -1.0,
        llama_cpp_yarn_attn_factor: float = 1.0,
        llama_cpp_yarn_beta_fast: float = 32.0,
        llama_cpp_yarn_beta_slow: float = 1.0,
        llama_cpp_yarn_orig_ctx: int = 0,
        llama_cpp_mul_mat_q: bool = True,
        llama_cpp_logits_all: bool = False,
        llama_cpp_embedding: bool = False,
        llama_cpp_offload_kqv: bool = True,
        llama_cpp_last_n_tokens_size: int = 64,
        llama_cpp_lora_base: Optional[str] = None,
        llama_cpp_lora_scale: float = 1.0,
        llama_cpp_lora_path: Optional[str] = None,
        llama_cpp_numa: Union[bool, int] = False,
        llama_cpp_chat_format: Optional[str] = None,
        llama_cpp_draft_model: Optional[llama_cpp.LlamaDraftModel] = None,
        # llama_cpp_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        llama_cpp_verbose: bool = True,
        # Server params
        endpoint: str = "*",
        port: int = 3000,
        cors_domain: str = "http://localhost:3000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **model_args: Any,
    ) -> None:
        """
        Starts a CherryPy server to listen for requests to generate text.

        Args:
            model_name (str): Name or identifier of the pre-trained model to be used.
            model_class (str): Class name of the model to be used from the transformers library.
            tokenizer_class (str): Class name of the tokenizer to be used from the transformers library.
            use_cuda (bool): Flag to enable CUDA for GPU acceleration.
            precision (str): Specifies the precision configuration for PyTorch tensors, e.g., "float16".
            quantization (int): Level of model quantization to reduce model size and inference time.
            device_map (Union[str, Dict, None]): Maps model layers to specific devices for distributed inference.
            max_memory (Dict[int, str]): Maximum memory allocation for the model on each device.
            torchscript (bool): Enables the use of TorchScript for model optimization.
            compile (bool): Enables model compilation for further optimization.
            awq_enabled (bool): Enables Adaptive Weight Quantization (AWQ) for model optimization.
            flash_attention (bool): Utilizes Flash Attention optimizations for faster processing.
            concurrent_queries (bool): Allows the server to handle multiple requests concurrently if True.
            use_vllm (bool): Flag to use Very Large Language Models (VLLM) integration.
            use_llama_cpp (bool): Flag to use llama.cpp integration for language model inference.
            llama_cpp_filename (Optional[str]): The filename of the model file for llama.cpp.
            llama_cpp_n_gpu_layers (int): Number of layers to offload to GPU in llama.cpp configuration.
            llama_cpp_split_mode (int): Defines how the model is split across multiple GPUs in llama.cpp.
            llama_cpp_tensor_split (Optional[List[float]]): Custom tensor split configuration for llama.cpp.
            llama_cpp_vocab_only (bool): Loads only the vocabulary part of the model in llama.cpp.
            llama_cpp_use_mmap (bool): Enables memory-mapped files for model loading in llama.cpp.
            llama_cpp_use_mlock (bool): Locks the model in RAM to prevent swapping in llama.cpp.
            llama_cpp_kv_overrides (Optional[Dict[str, Union[bool, int, float]]]): Key-value pairs for overriding default llama.cpp model parameters.
            llama_cpp_seed (int): Seed for random number generation in llama.cpp.
            llama_cpp_n_ctx (int): The number of context tokens for the model in llama.cpp.
            llama_cpp_n_batch (int): Batch size for processing prompts in llama.cpp.
            llama_cpp_n_threads (Optional[int]): Number of threads for generation in llama.cpp.
            llama_cpp_n_threads_batch (Optional[int]): Number of threads for batch processing in llama.cpp.
            llama_cpp_rope_scaling_type (Optional[int]): Specifies the RoPE (Rotary Positional Embeddings) scaling type in llama.cpp.
            llama_cpp_rope_freq_base (float): Base frequency for RoPE in llama.cpp.
            llama_cpp_rope_freq_scale (float): Frequency scaling factor for RoPE in llama.cpp.
            llama_cpp_yarn_ext_factor (float): Extrapolation mix factor for YaRN in llama.cpp.
            llama_cpp_yarn_attn_factor (float): Attention factor for YaRN in llama.cpp.
            llama_cpp_yarn_beta_fast (float): Beta fast parameter for YaRN in llama.cpp.
            llama_cpp_yarn_beta_slow (float): Beta slow parameter for YaRN in llama.cpp.
            llama_cpp_yarn_orig_ctx (int): Original context size for YaRN in llama.cpp.
            llama_cpp_mul_mat_q (bool): Flag to enable matrix multiplication for queries in llama.cpp.
            llama_cpp_logits_all (bool): Returns logits for all tokens when set to True in llama.cpp.
            llama_cpp_embedding (bool): Enables embedding mode only in llama.cpp.
            llama_cpp_offload_kqv (bool): Offloads K, Q, V matrices to GPU in llama.cpp.
            llama_cpp_last_n_tokens_size (int): Size for the last_n_tokens buffer in llama.cpp.
            llama_cpp_lora_base (Optional[str]): Base model path for LoRA adjustments in llama.cpp.
            llama_cpp_lora_scale (float): Scale factor for LoRA adjustments in llama.cpp.
            llama_cpp_lora_path (Optional[str]): Path to LoRA adjustments file in llama.cpp.
            llama_cpp_numa (Union[bool, int]): NUMA configuration for llama.cpp.
            llama_cpp_chat_format (Optional[str]): Specifies the chat format for llama.cpp.
            llama_cpp_draft_model (Optional[llama_cpp.LlamaDraftModel]): Draft model for speculative decoding in llama.cpp.
            endpoint (str): Network interface to bind the server to.
            port (int): Port number to listen on for incoming requests.
            cors_domain (str): Specifies the domain to allow for Cross-Origin Resource Sharing (CORS).
            username (Optional[str]): Username for basic authentication, if required.
            password (Optional[str]): Password for basic authentication, if required.
            **model_args (Any): Additional arguments to pass to the pre-trained language model or llama.cpp configuration.
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
        self.awq_enabled = awq_enabled
        self.flash_attention = flash_attention
        self.use_vllm = use_vllm
        self.concurrent_queries = concurrent_queries

        self.model_args = model_args
        self.username = username
        self.password = password

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

        if use_vllm:
            self.model = self.load_models_vllm(
                model=model_name,
                tokenizer=tokenizer_name,
                tokenizer_mode=vllm_tokenizer_mode,
                trust_remote_code=True,
                download_dir=vllm_download_dir,
                load_format=vllm_load_format,
                dtype=self._get_torch_dtype(precision),
                seed=vllm_seed,
                revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                max_model_len=vllm_max_model_len,
                quantization=(None if quantization == 0 else f"{quantization}-bit"),
                enforce_eager=vllm_enforce_eager,
                max_context_len_to_capture=vllm_max_context_len_to_capture,
                block_size=vllm_block_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                swap_space=vllm_swap_space,
                cache_dtype="auto",
                sliding_window=vllm_sliding_window,
                pipeline_parallel_size=vllm_pipeline_parallel_size,
                tensor_parallel_size=vllm_tensor_parallel_size,
                worker_use_ray=vllm_worker_use_ray,
                max_parallel_loading_workers=vllm_max_parallel_loading_workers,
                disable_custom_all_reduce=vllm_disable_custom_all_reduce,
                max_num_batched_tokens=vllm_max_num_batched_tokens,
                max_num_seqs=vllm_max_num_seqs,
                max_paddings=vllm_max_paddings,
                device="cuda" if use_cuda else "cpu",
                max_lora_rank=vllm_max_lora_rank,
                max_loras=vllm_max_loras,
                max_cpu_loras=vllm_max_cpu_loras,
                lora_dtype=self._get_torch_dtype(precision),
                lora_extra_vocab_size=vllm_lora_extra_vocab_size,
                placement_group=vllm_placement_group,  # type: ignore
                log_stats=vllm_log_stats,
                batched_inference=False,
            )
        elif use_llama_cpp:
            self.model, self.tokenizer = self.load_models_llama_cpp(
                model=self.model_name,
                filename=llama_cpp_filename,
                local_dir=self.output.output_folder,
                n_gpu_layers=llama_cpp_n_gpu_layers,
                split_mode=llama_cpp_split_mode,
                main_gpu=0 if self.use_cuda else -1,
                tensor_split=llama_cpp_tensor_split,
                vocab_only=llama_cpp_vocab_only,
                use_mmap=llama_cpp_use_mmap,
                use_mlock=llama_cpp_use_mlock,
                kv_overrides=llama_cpp_kv_overrides,
                seed=llama_cpp_seed,
                n_ctx=llama_cpp_n_ctx,
                n_batch=llama_cpp_n_batch,
                n_threads=llama_cpp_n_threads,
                n_threads_batch=llama_cpp_n_threads_batch,
                rope_scaling_type=llama_cpp_rope_scaling_type,
                rope_freq_base=llama_cpp_rope_freq_base,
                rope_freq_scale=llama_cpp_rope_freq_scale,
                yarn_ext_factor=llama_cpp_yarn_ext_factor,
                yarn_attn_factor=llama_cpp_yarn_attn_factor,
                yarn_beta_fast=llama_cpp_yarn_beta_fast,
                yarn_beta_slow=llama_cpp_yarn_beta_slow,
                yarn_orig_ctx=llama_cpp_yarn_orig_ctx,
                mul_mat_q=llama_cpp_mul_mat_q,
                logits_all=llama_cpp_logits_all,
                embedding=llama_cpp_embedding,
                offload_kqv=llama_cpp_offload_kqv,
                last_n_tokens_size=llama_cpp_last_n_tokens_size,
                lora_base=llama_cpp_lora_base,
                lora_scale=llama_cpp_lora_scale,
                lora_path=llama_cpp_lora_path,
                numa=llama_cpp_numa,
                chat_format=llama_cpp_chat_format,
                draft_model=llama_cpp_draft_model,
                # tokenizer=llama_cpp_tokenizer, # TODO: support custom tokenizers for llama.cpp
                verbose=llama_cpp_verbose,
                **model_args,
            )
        else:
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

        def sequential_locker():
            if self.concurrent_queries:
                sequential_lock.acquire()

        def sequential_unlocker():
            if self.concurrent_queries:
                sequential_lock.release()

        def CORS():
            cherrypy.response.headers["Access-Control-Allow-Origin"] = cors_domain
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
                "error_page.400": error_page,
                "error_page.401": error_page,
                "error_page.402": error_page,
                "error_page.403": error_page,
                "error_page.404": error_page,
                "error_page.405": error_page,
                "error_page.406": error_page,
                "error_page.408": error_page,
                "error_page.415": error_page,
                "error_page.429": error_page,
                "error_page.500": error_page,
                "error_page.501": error_page,
                "error_page.502": error_page,
                "error_page.503": error_page,
                "error_page.504": error_page,
                "error_page.default": error_page,
            }
        )

        if username and password:
            # Configure basic authentication
            conf = {
                "/": {
                    "tools.sequential_locker.on": True,
                    "tools.sequential_unlocker.on": True,
                    "tools.auth_basic.on": True,
                    "tools.auth_basic.realm": "geniusrise",
                    "tools.auth_basic.checkpassword": self.validate_password,
                    "tools.CORS.on": True,
                }
            }
        else:
            # Configuration without authentication
            conf = {
                "/": {
                    "tools.sequential_locker.on": True,
                    "tools.sequential_unlocker.on": True,
                    "tools.CORS.on": True,
                }
            }

        cherrypy.tools.sequential_locker = cherrypy.Tool("before_handler", sequential_locker)
        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", conf)
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.tools.sequential_unlocker = cherrypy.Tool("before_finalize", sequential_unlocker)
        cherrypy.engine.start()
        cherrypy.engine.block()


def error_page(status, message, traceback, version):
    response = {
        "status": status,
        "message": message,
    }
    return json.dumps(response)
