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


class SummarizationBulk(TextBulk):
    r"""
    SummarizationBulk is a class for managing bulk text summarization tasks using Hugging Face models. It is
    designed to handle large-scale summarization tasks efficiently and effectively, utilizing state-of-the-art
    machine learning models to provide high-quality summaries.

    The class provides methods to load datasets, configure summarization models, and execute bulk summarization tasks.

    Example CLI Usage:
    ```bash
    genius SummarizationBulk rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder input/summz \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder output/summz \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise\
            --postgres_table state \
        --id facebook/bart-large-cnn-lol \
        summarize \
            --args \
                model_name="facebook/bart-large-cnn" \
                model_class="AutoModelForSeq2SeqLM" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="float" \
                quantization=0 \
                device_map="cuda:0" \
                max_memory=None \
                torchscript=False \
                generation_bos_token_id=0 \
                generation_decoder_start_token_id=2 \
                generation_early_stopping=true \
                generation_eos_token_id=2 \
                generation_forced_bos_token_id=0 \
                generation_forced_eos_token_id=2 \
                generation_length_penalty=2.0 \
                generation_max_length=142 \
                generation_min_length=56 \
                generation_no_repeat_ngram_size=3 \
                generation_num_beams=4 \
                generation_pad_token_id=1 \
                generation_do_sample=false
    ```
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        """
        Initializes the SummarizationBulk class.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        r"""
        Load a dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset | DatasetDict: The loaded dataset.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content"}
        ```

        ### CSV
        Should contain a 'text' column.
        ```csv
        text
        "The text content"
        ```

        ### Parquet
        Should contain a 'text' column.

        ### JSON
        An array of dictionaries with a 'text' key.
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
        Each document should be a dictionary with a 'text' key.
        ```yaml
        - text: "The text content"
        ```

        ### TSV
        Should contain a 'text' column.

        ### Excel (.xls, .xlsx)
        Should contain a 'text' column.

        ### SQLite (.db)
        Should contain a table with a 'text' column.

        ### Feather
        Should contain a 'text' column.
        """
        try:
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                dataset = load_from_disk(dataset_path)
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

                dataset = Dataset.from_pandas(pd.DataFrame(data))
            return dataset

        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def summarize(
        self,
        model_name: str,
        model_class: str = "AutoModelForSeq2SeqLM",
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
        batch_size: int = 32,
        max_length: int = 512,
        notification_email: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Perform bulk summarization using the specified model and tokenizer. This method handles the entire summarization
        process including loading the model, processing input data, generating summarization, and saving the results.

        Args:
            model_name (str): Name or path of the translation model.
            origin (str): Source language ISO code.
            target (str): Target language ISO code.
            max_length (int): Maximum length of the tokens (default 512).
            model_class (str): Class name of the model (default "AutoModelForSeq2SeqLM").
            tokenizer_class (str): Class name of the tokenizer (default "AutoTokenizer").
            use_cuda (bool): Whether to use CUDA for model inference (default False).
            precision (str): Precision for model computation (default "float16").
            quantization (int): Level of quantization for optimizing model size and speed (default 0).
            device_map (str | Dict | None): Specific device to use for computation (default "auto").
            max_memory (Dict): Maximum memory configuration for devices.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            awq_enabled (bool): Whether to enable AWQ optimization (default False).
            flash_attention (bool): Whether to use flash attention optimization (default False).
            batch_size (int): Number of translations to process simultaneously (default 32).
            max_length (int): Maximum lenght of the summary to be generated (default 512).
            **kwargs: Arbitrary keyword arguments for model and generation configurations.
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
        self.batch_size = batch_size
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
        _dataset = self.load_dataset(dataset_path, max_lengt=max_length)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["text"]

        # Process data in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate summaries
            summaries = self.model.generate(**inputs, **self.generation_args)
            if next(self.model.parameters()).is_cuda:
                summaries = summaries.cpu()

            decoded_summaries = [self.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]

            self._save_summaries(decoded_summaries, batch, output_path, i)
        self.done()

    def _save_summaries(self, summaries: List[str], input_batch: List[str], output_path: str, batch_idx: int) -> None:
        """
        Saves the generated summaries to a specified output path. This method is called internally by the summarize method
        to persist the summarization results.

        Args:
            summaries (List[str]): List of generated summaries.
            input_batch (List[str]): List of original texts that were summarized.
            output_path (str): Path to save the summaries.
            batch_idx (int): Index of the current batch (for naming files).
        """
        data_to_save = [
            {"input": input_text, "summary": summary} for input_text, summary in zip(input_batch, summaries)
        ]
        with open(os.path.join(output_path, f"summaries-{batch_idx}-{str(uuid.uuid4())}.json"), "w") as f:
            json.dump(data_to_save, f)
