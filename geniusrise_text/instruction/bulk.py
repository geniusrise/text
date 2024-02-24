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

import glob
import json
import os
import sqlite3
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, load_from_disk
from geniusrise import BatchInput, BatchOutput, State
from pyarrow import feather
from pyarrow import parquet as pq
from vllm import LLM, SamplingParams

from geniusrise_text.base import TextBulk


class InstructionBulk(TextBulk):
    r"""
    InstructionBulk is a class designed to perform bulk text generation tasks using Hugging Face's instruction-tuned language models.
    It is optimized for large-scale text generation, providing an efficient interface to use state-of-the-art machine learning
    models for generating text based on a set of instructions or prompts.

    Attributes:
        model (Any): The loaded, pre-trained instruction-tuned language model.
        tokenizer (Any): The tokenizer for processing text compatible with the model.

    Methods:
        load_dataset(dataset_path: str, max_length: int = 1024, **kwargs) -> Optional[Dataset]:
            Loads a dataset for text generation tasks from the specified directory.

        perform(model_name: str, **kwargs: Any) -> None:
            Performs bulk text generation using the specified model and tokenizer.

    Example CLI Usage:
    ```bash
    genius InstructionBulk rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder input/chat \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder output/chat \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise\
            --postgres_table state \
        --id mistralai/Mistral-7B-Instruct-v0.1-lol \
        perform \
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
                decoding_strategy="generate" \
                generation_max_new_tokens=100 \
                generation_do_sample=true
    ```

    or via VLLM:
    ```bash
    genius InstructionBulk rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder input/chat \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder output/chat \
        none \
        --id mistralai/Mistral-7B-Instruct-v0.1 \
        perform_vllm \
            --args \
                model_name="mistralai/Mistral-7B-Instruct-v0.1" \
                use_cuda=True \
                precision="bfloat16" \
                quantization=0 \
                device_map="auto" \
                generation_temperature=0.7 \
                generation_top_p=1.0 \
                generation_n=1 \
                generation_max_tokens=50 \
                generation_stream=false \
                generation_presence_penalty=0.0 \
                generation_frequency_penalty=0.0
    ```
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        """
        Initializes the InstructionBulk class with input, output, and state configurations for bulk text generation.

        Args:
            input (BatchInput): Configuration for input data handling.
            output (BatchOutput): Configuration for output data handling.
            state (State): State management for the text generation task.
            **kwargs: Additional keyword arguments for extended functionalities.
        """
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 1024, **kwargs) -> Optional[Dataset]:
        r"""
        Loads a dataset from the specified path. This method supports various data formats including JSON, CSV, Parquet,
        and others. It's designed to facilitate the bulk processing of text data for generation tasks.

        Args:
            dataset_path (str): Path to the directory containing the dataset files.
            max_length (int): Maximum token length for text processing (default is 1024).
            **kwargs: Additional keyword arguments for dataset loading.

        Returns:
            Optional[Dataset]: A Dataset object if loading is successful; otherwise, None.

        Raises:
            Exception: If an error occurs during dataset loading.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"instruction": "The instruction"}
        ```

        ### CSV
        Should contain 'instruction' columns.
        ```csv
        instruction
        "The instruction"
        ```

        ### Parquet
        Should contain 'instruction' columns.

        ### JSON
        An array of dictionaries with 'instruction' keys.
        ```json
        [{"instruction": "The instruction"}]
        ```

        ### XML
        Each 'record' element should contain 'instruction' child elements.
        ```xml
        <record>
            <instruction>The instruction</instruction>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'instruction' keys.
        ```yaml
        - instruction: "The instruction"
        ```

        ### TSV
        Should contain 'instruction' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'instruction' columns.

        ### SQLite (.db)
        Should contain a table with 'instruction' columns.

        ### Feather
        Should contain 'instruction' columns.
        """
        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            self.max_length = max_length
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                return load_from_disk(dataset_path)
            else:
                data = []
                for filename in glob.glob(f"{dataset_path}/**/*", recursive=True):
                    filepath = os.path.join(dataset_path, filename)
                    if filename.endswith(".jsonl"):
                        with open(filepath, "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)

                    elif filename.endswith(".csv"):
                        df = pd.read_csv(filepath)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".parquet"):
                        df = pq.read_table(filepath).to_pandas()
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            json_data = json.load(f)
                            data.extend(json_data)

                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            instruction = record.find("instruction").text  # type: ignore
                            data.append({"instruction": instruction})

                    elif filename.endswith(".yaml") or filename.endswith(".yml"):
                        with open(filepath, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            data.extend(yaml_data)

                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))

                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT instruction FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                if hasattr(self, "map_data") and self.map_data:
                    fn = eval(self.map_data)  # type: ignore
                    data = [fn(d) for d in data]
                else:
                    data = data

                dataset = Dataset.from_pandas(pd.DataFrame(data))
                return dataset
        except Exception as e:
            self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def perform(
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
        compile: bool = False,
        awq_enabled: bool = False,
        flash_attention: bool = False,
        decoding_strategy: str = "generate",
        notification_email: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Performs text generation in bulk using a specified instruction-tuned model. This method handles the entire
        process, including model loading, prompt processing, text generation, and saving the results.

        Args:
            model_name (str): The name or path of the instruction-tuned model.
            model_class (str, optional): The class of the language model. Defaults to "AutoModelForCausalLM".
            tokenizer_class (str, optional): The class of the tokenizer. Defaults to "AutoTokenizer".
            use_cuda (bool, optional): Whether to use CUDA for model inference. Defaults to False.
            precision (str, optional): Precision for model computation. Defaults to "float16".
            quantization (int, optional): Level of quantization for optimizing model size and speed. Defaults to 0.
            device_map (str | Dict | None, optional): Specific device to use for computation. Defaults to "auto".
            max_memory (Dict, optional): Maximum memory configuration for devices. Defaults to {0: "24GB"}.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            awq_enabled (bool, optional): Whether to enable AWQ optimization. Defaults to False.
            flash_attention (bool, optional): Whether to use flash attention optimization. Defaults to False.
            decoding_strategy (str, optional): Strategy for decoding the completion. Defaults to "generate".
            **kwargs: Configuration and additional arguments for text generation such as model class, tokenizer class,
                      precision, device map, and other generation-related parameters.

        Note:
            Additional arguments are passed directly to the model and tokenizer initialization and the generation method.
        """
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
        self.tokenizer_name = tokenizer_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.awq_enabled = awq_enabled
        self.flash_attention = flash_attention
        self.notification_email = notification_email
        self.compile = compile

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

        generation_args = {k.replace("generation_", ""): v for k, v in kwargs.items() if "generation_" in k}
        self.generation_args = generation_args

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
            compile=self.compile,
            **self.model_args,
        )

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        _dataset = self.load_dataset(dataset_path)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["instruction"]

        prompts = []
        completions = []

        for _, prompt in enumerate(dataset):
            completion = self.generate(
                prompt=prompt,
                decoding_strategy=decoding_strategy,
                **generation_args,
            )
            completions.append(completion)
            prompts.append(prompt)

        self._save_completions(completions, prompts, output_path)
        self.done()

    def perform_vllm(
        self,
        model_name: str,
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
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
        # Generate params
        notification_email: Optional[str] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """
        Performs bulk text generation using the Versatile Language Learning Model (VLLM) with specified parameters
        for fine-tuning model behavior, including quantization and parallel processing settings. This method is designed
        to process large datasets efficiently by leveraging VLLM capabilities for generating high-quality text completions
        based on provided prompts.

        Args:
            model_name (str): The name or path of the VLLM model to use for text generation.
            use_cuda (bool): Flag indicating whether to use CUDA for GPU acceleration.
            precision (str): Precision of computations, can be "float16", "bfloat16", etc.
            quantization (int): Level of quantization for model weights, 0 for none.
            device_map (str | Dict | None): Specific device(s) to use for model inference.
            vllm_tokenizer_mode (str): Mode of the tokenizer ("auto", "fast", or "slow").
            vllm_download_dir (Optional[str]): Directory to download and load the model and tokenizer.
            vllm_load_format (str): Format to load the model, e.g., "auto", "pt".
            vllm_seed (int): Seed for random number generation.
            vllm_max_model_len (int): Maximum sequence length the model can handle.
            vllm_enforce_eager (bool): Enforce eager execution instead of using optimization techniques.
            vllm_max_context_len_to_capture (int): Maximum context length for CUDA graph capture.
            vllm_block_size (int): Block size for caching mechanism.
            vllm_gpu_memory_utilization (float): Fraction of GPU memory to use.
            vllm_swap_space (int): Amount of swap space to use in GiB.
            vllm_sliding_window (Optional[int]): Size of the sliding window for processing.
            vllm_pipeline_parallel_size (int): Number of pipeline parallel groups.
            vllm_tensor_parallel_size (int): Number of tensor parallel groups.
            vllm_worker_use_ray (bool): Whether to use Ray for model workers.
            vllm_max_parallel_loading_workers (Optional[int]): Maximum number of workers for parallel loading.
            vllm_disable_custom_all_reduce (bool): Disable custom all-reduce kernel and fall back to NCCL.
            vllm_max_num_batched_tokens (Optional[int]): Maximum number of tokens to be processed in a single iteration.
            vllm_max_num_seqs (int): Maximum number of sequences to be processed in a single iteration.
            vllm_max_paddings (int): Maximum number of paddings to be added to a batch.
            vllm_max_lora_rank (Optional[int]): Maximum rank for LoRA adjustments.
            vllm_max_loras (Optional[int]): Maximum number of LoRA adjustments.
            vllm_max_cpu_loras (Optional[int]): Maximum number of LoRA adjustments stored on CPU.
            vllm_lora_extra_vocab_size (int): Additional vocabulary size for LoRA.
            vllm_placement_group (Optional[dict]): Ray placement group for distributed execution.
            vllm_log_stats (bool): Whether to log statistics during model operation.
            notification_email (Optional[str]): Email to send notifications upon completion.
            batch_size (int): Number of prompts to process in each batch for efficient memory usage.
            **kwargs: Additional keyword arguments for generation settings like temperature, top_p, etc.

        This method automates the loading of large datasets, generation of text completions, and saving results,
        facilitating efficient and scalable text generation tasks.
        """
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
        self.tokenizer_name = tokenizer_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.notification_email = notification_email

        self.model: LLM = self.load_models_vllm(
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
            batched_inference=True,
        )

        generation_args = {k.replace("generation_", ""): v for k, v in kwargs.items() if "generation_" in k}
        self.generation_args = generation_args

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        _dataset = self.load_dataset(dataset_path)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["instruction"]

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            outputs = self.model.generate(
                prompts=batch,
                sampling_params=SamplingParams(
                    n=generation_args.get("n", 1),
                    best_of=generation_args.get("best_of", None),
                    presence_penalty=generation_args.get("presence_penalty", 0.0),
                    frequency_penalty=generation_args.get("frequency_penalty", 0.0),
                    repetition_penalty=generation_args.get("repetition_penalty", 1.0),
                    temperature=generation_args.get("temperature", 1.0),
                    top_p=generation_args.get("top_p", 1.0),
                    top_k=generation_args.get("top_k", -1),
                    min_p=generation_args.get("min_p", 0.0),
                    use_beam_search=generation_args.get("use_beam_search", False),
                    length_penalty=generation_args.get("length_penalty", 1.0),
                    early_stopping=generation_args.get("early_stopping", False),
                    stop=generation_args.get("stop", None),
                    stop_token_ids=generation_args.get("stop_token_ids", None),
                    include_stop_str_in_output=generation_args.get("include_stop_str_in_output", False),
                    ignore_eos=generation_args.get("ignore_eos", False),
                    max_tokens=generation_args.get("max_tokens", 16),
                    logprobs=generation_args.get("logprobs", None),
                    prompt_logprobs=generation_args.get("prompt_logprobs", None),
                    skip_special_tokens=generation_args.get("skip_special_tokens", True),
                    spaces_between_special_tokens=generation_args.get("spaces_between_special_tokens", True),
                    logits_processors=generation_args.get("logits_processors", None),
                ),
            )
            completions = [" ".join(t.text for t in o.outputs) for o in outputs]
            self._save_completions(completions, batch, output_path)
        self.done()

    def _save_completions(self, completions: List[str], prompts: List[str], output_path: str) -> None:
        """
        Saves the generated texts alongside their prompts to the specified output path. This method ensures the results
        of text generation are persisted for later use or analysis.

        Args:
            completions (List[str]): The list of generated texts.
            prompts (List[str]): The list of prompts corresponding to the generated texts.
            output_path (str): The directory path to save the results.
        """
        data_to_save = [
            {"prompt": prompt, "completion": completion} for prompt, completion in zip(prompts, completions)
        ]
        with open(os.path.join(output_path, f"completions-{str(uuid.uuid4())}.json"), "w") as f:
            json.dump(data_to_save, f)
