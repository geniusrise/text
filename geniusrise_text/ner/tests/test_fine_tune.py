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
import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import torch
import yaml  # type: ignore
from datasets import Dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import EvalPrediction

from geniusrise_text.ner.fine_tune import NamedEntityRecognitionFineTuner

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [{"tokens": ["This", "is", "a", "test"], "ner_tags": [0, 1, 0, 1]} for _ in range(10)]
    df = pd.DataFrame(data)

    if ext == "huggingface":
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(directory)
    elif ext == "csv":
        df.to_csv(os.path.join(directory, "data.csv"), index=False)
    elif ext == "jsonl":
        with open(os.path.join(directory, "data.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif ext == "parquet":
        pq.write_table(feather.Table.from_pandas(df), os.path.join(directory, "data.parquet"))
    elif ext == "json":
        with open(os.path.join(directory, "data.json"), "w") as f:
            json.dump(data, f)
    elif ext == "xml":
        root = ET.Element("root")
        for item in data:
            record = ET.SubElement(root, "record")
            ET.SubElement(record, "tokens").text = " ".join(item["tokens"])
            ET.SubElement(record, "ner_tags").text = " ".join(map(str, item["ner_tags"]))
        tree = ET.ElementTree(root)
        tree.write(os.path.join(directory, "data.xml"))
    elif ext == "yaml":
        with open(os.path.join(directory, "data.yaml"), "w") as f:
            yaml.dump(data, f)
    elif ext == "tsv":
        df.to_csv(os.path.join(directory, "data.tsv"), index=False, sep="\t")
    elif ext == "xlsx":
        df.to_excel(os.path.join(directory, "data.xlsx"), index=False)
    elif ext == "db":
        conn = sqlite3.connect(os.path.join(directory, "data.db"))
        df.to_sql("dataset_table", conn, if_exists="replace", index=False)
        conn.close()
    elif ext == "feather":
        feather.write_feather(df, os.path.join(directory, "data.feather"))


# Fixtures for each file type
@pytest.fixture(
    params=[
        "huggingface",
        "csv",
        "jsonl",
        "parquet",
        "json",
        "xml",
        "yaml",
        "tsv",
        "xlsx",
        "feather",
    ]
)
def dataset_file(request, tmpdir):
    ext = request.param
    create_dataset_in_format(tmpdir + "/train", ext)
    create_dataset_in_format(tmpdir + "/eval", ext)
    return tmpdir, ext


MODELS_TO_TEST = {
    # fmt: off
    "small": "dslim/bert-large-NER",
    # fmt: on
}


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


@pytest.fixture
def ner_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()

    klass = NamedEntityRecognitionFineTuner(
        input=input,
        output=output,
        state=state,
    )
    klass.model_class = "BertForTokenClassification"
    klass.model_name = "bert-base-uncased"
    klass.tokenizer_class = "BertTokenizerFast"
    klass.tokenizer_name = "bert-base-uncased"

    return klass


def test_ner_bolt_init(ner_bolt, model):
    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForTokenClassification"
    tokenizer_class = "AutoTokenizer"

    ner_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    assert ner_bolt.model is not None
    assert ner_bolt.tokenizer is not None
    assert ner_bolt.input is not None
    assert ner_bolt.output is not None
    assert ner_bolt.state is not None


def test_load_dataset_all_formats(ner_bolt, dataset_file, model):
    tmpdir, ext = dataset_file
    dataset_path = os.path.join(tmpdir, "train")

    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForTokenClassification"
    tokenizer_class = "AutoTokenizer"

    ner_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )
    dataset = ner_bolt.load_dataset(dataset_path, label_list=[0, 1])
    assert dataset is not None
    assert len(dataset) == 10


# Models to test
models = {
    # fmt: off
    "bart": "dslim/bert-large-NER",
    "wikineural": "Babelscape/wikineural-multilingual-ner",
    "medical": "d4data/biomedical-ner-all",
    "chemical": "alvaroalon2/biobert_chemical_ner",
    "genetic": "pruas/BENT-PubMedBERT-NER-Gene",
    "food": "Dizex/FoodBaseBERT-NER",
    "disease": "pruas/BENT-PubMedBERT-NER-Disease",
    # fmt: on
}


# Test for fine-tuning
@pytest.mark.parametrize(
    "model_name, precision, quantization, lora_config, use_accelerate",
    [
        # small
        (models["bart"], "bfloat16", None, None, False),
        (models["wikineural"], "bfloat16", None, None, False),
        (models["medical"], "bfloat16", None, None, False),
        (models["chemical"], "bfloat16", None, None, False),
        (models["genetic"], "bfloat16", None, None, False),
        (models["food"], "bfloat16", None, None, False),
        (models["disease"], "bfloat16", None, None, False),
    ],
)
def test_ner_bolt_fine_tune(ner_bolt, dataset_file, model_name, precision, quantization, lora_config, use_accelerate):
    try:
        tokenizer_name = model_name

        tmpdir, ext = dataset_file
        ner_bolt.input.input_folder = tmpdir

        ner_bolt.fine_tune(
            model_name=model_name,
            tokenizer_name=model_name,
            model_class="AutoModelForTokenClassification",
            tokenizer_class="AutoTokenizer",
            num_train_epochs=1,
            per_device_batch_size=2,
            precision=precision,
            quantization=quantization,
            lora_config=lora_config,
            use_accelerate=use_accelerate,
            device_map="cuda:0",
            data_label_list=[0, 1],
        )
        output_dir = ner_bolt.output.output_folder
        assert os.path.exists(
            os.path.join(ner_bolt.output.output_folder, "model", "pytorch_model.bin")
        ) or os.path.exists(os.path.join(ner_bolt.output.output_folder, "model", "adapter_model.bin"))
        assert os.path.exists(os.path.join(ner_bolt.output.output_folder, "model", "config.json")) or os.path.exists(
            os.path.join(ner_bolt.output.output_folder, "model", "adapter_config.json")
        )
        assert os.path.exists(os.path.join(ner_bolt.output.output_folder, "model", "training_args.bin"))

        del ner_bolt.model
        del ner_bolt.tokenizer
        torch.cuda.empty_cache()

        try:
            os.remove(os.path.join(ner_bolt.output.output_folder, "model", "pytorch_model.bin"))
            os.remove(os.path.join(ner_bolt.output.output_folder, "model", "adapter_model.bin"))
            os.remove(os.path.join(ner_bolt.output.output_folder, "model", "config.json"))
            os.remove(os.path.join(ner_bolt.output.output_folder, "model", "adapter_config.json"))
            os.remove(os.path.join(ner_bolt.output.output_folder, "model", "training_args.bin"))
        except Exception as _:
            pass

    except Exception as e:
        del ner_bolt.model
        del ner_bolt.tokenizer
        torch.cuda.empty_cache()
        raise


# Test for computing metrics
def test_ner_bolt_compute_metrics(ner_bolt):
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    metrics = ner_bolt.compute_metrics(eval_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
