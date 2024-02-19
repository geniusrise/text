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

from geniusrise_text.base import TextBulk


class LanguageModelBulk(TextBulk):
    r"""
    LanguageModelBulk is designed for large-scale text generation using Hugging Face language models in a bulk processing
    manner. It's particularly useful for tasks such as bulk content creation, summarization, or any other scenario where
    large datasets need to be processed with a language model.

    Attributes:
        model (Any): The loaded language model used for text generation.
        tokenizer (Any): The tokenizer corresponding to the language model, used for processing input text.

    Args:
        input (BatchInput): Configuration for the input data.
        output (BatchOutput): Configuration for the output data.
        state (State): State management for the API.
        **kwargs (Any): Arbitrary keyword arguments for extended functionality.

    CLI Usage Example:
    ```bash
    genius LanguageModelBulk rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder input/lm \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder output/lm \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise\
            --postgres_table state \
        --id mistralai/Mistral-7B-Instruct-v0.1-lol \
        complete \
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
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        """
        Initializes the LanguageModelBulk object with the specified configurations for input, output, and state.

        Args:
            input (BatchInput): Configuration and data inputs for the bulk process.
            output (BatchOutput): Configurations for output data handling.
            state (State): State management for the bulk process.
            **kwargs (Any): Additional keyword arguments for extended configurations.
        """
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        r"""
        Load a completion dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.
            **kwargs: Additional keyword arguments to pass to the underlying dataset loading functions.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Dataset files saved by Hugging Face datasets library
        The directory should contain 'dataset_info.json' and other related files.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content"}
        ```

        ### CSV
        Should contain 'text' column.
        ```csv
        text
        "The text content"
        ```

        ### Parquet
        Should contain 'text' column.

        ### JSON
        An array of dictionaries with 'text' key.
        ```json
        [{"text": "The text content"}]
        ```

        ### XML
        Each 'record' element should contain 'text' child element.
        ```xml
        <record>
            <text>The text content</text>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' key.
        ```yaml
        - text: "The text content"
        ```

        ### TSV
        Should contain 'text' column separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' column.

        ### SQLite (.db)
        Should contain a table with 'text' column.

        ### Feather
        Should contain 'text' column.
        """

        self.max_length = max_length

        self.label_to_id = self.model.config.label2id if self.model and self.model.config.label2id else {}  # type: ignore

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
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
                            text = record.find("text").text  # type: ignore
                            data.append({"text": text})

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
                        query = "SELECT text FROM dataset_table;"
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

                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def complete(
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
        Performs text completion on the loaded dataset using the specified model and tokenizer. The method handles the
        entire process, including model loading, text generation, and saving the results.

        Args:
            model_name (str): The name of the language model to use for text completion.
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
            **kwargs: Additional keyword arguments for text generation.
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
        dataset = _dataset["text"]

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

    def _save_completions(self, completions: List[str], prompts: List[str], output_path: str) -> None:
        """
        Saves the generated completions to the specified output path.

        Args:
            completions (List[str]): The list of generated text completions.
            prompts (List[str]): The list of prompts corresponding to the completions.
            output_path (str): The path to save the completion results.

        This method is called internally by the complete method to persist the completion results.
        """
        data_to_save = [
            {"prompt": prompt, "completion": completion} for prompt, completion in zip(prompts, completions)
        ]
        with open(os.path.join(output_path, f"completions-{str(uuid.uuid4())}.json"), "w") as f:
            json.dump(data_to_save, f)
