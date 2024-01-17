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

from geniusrise_text.nli.fine_tune import NLIFineTuner

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
    data = [{"premise": f"premise_{i}", "hypothesis": f"hypothesis_{i}", "label": i % 2} for i in range(10)]
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
            ET.SubElement(record, "premise").text = item["premise"]
            ET.SubElement(record, "hypothesis").text = item["hypothesis"]
            ET.SubElement(record, "label").text = str(item["label"])
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
        "db",
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
    "small": "facebook/bart-large-mnli",
    # fmt: on
}


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


@pytest.fixture
def commonsense_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()
    klass = NLIFineTuner(
        input=input,
        output=output,
        state=state,
    )
    return klass


def test_commonsense_bolt_init(commonsense_bolt, model):
    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForSequenceClassification"
    tokenizer_class = "AutoTokenizer"

    commonsense_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    assert commonsense_bolt.model is not None
    assert commonsense_bolt.tokenizer is not None
    assert commonsense_bolt.input is not None
    assert commonsense_bolt.output is not None
    assert commonsense_bolt.state is not None


def test_load_dataset_all_formats(commonsense_bolt, dataset_file, model):
    tmpdir, ext = dataset_file
    dataset_path = os.path.join(tmpdir, "train")

    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForSequenceClassification"
    tokenizer_class = "AutoTokenizer"

    commonsense_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    dataset = commonsense_bolt.load_dataset(dataset_path)
    assert dataset is not None
    assert len(dataset) == 10


# Models to test
models = {
    # fmt: off
    "bart": "facebook/bart-large-mnli",
    "deberta": "microsoft/deberta-v2-xlarge-mnli",
    "large": "khalidalt/DeBERTa-v3-large-mnli",
    "distill": "typeform/distilbert-base-uncased-mnli",
    "roberta": "roberta-large-mnli",
    "biggest": "microsoft/deberta-v2-xxlarge-mnli",
    "tasksource": "sileod/deberta-v3-large-tasksource-nli",
    "deberta-v3-large": "cross-encoder/nli-deberta-v3-large",
    # fmt: on
}


# Test for fine-tuning
@pytest.mark.parametrize(
    "model_name, precision, quantization, lora_config, use_accelerate",
    [
        # small
        (models["bart"], "bfloat16", None, None, False),
        (models["deberta"], "bfloat16", None, None, False),
        (models["large"], "bfloat16", None, None, False),
        (models["distill"], "bfloat16", None, None, False),
        (models["roberta"], "bfloat16", None, None, False),
        (models["biggest"], "bfloat16", None, None, False),
        (models["tasksource"], "bfloat16", None, None, False),
        (models["deberta-v3-large"], "bfloat16", None, None, False),
    ],
)
def test_commonsense_bolt_fine_tune(
    commonsense_bolt, dataset_file, model_name, precision, quantization, lora_config, use_accelerate
):
    try:
        tokenizer_name = model_name

        tmpdir, ext = dataset_file
        commonsense_bolt.input.input_folder = tmpdir

        commonsense_bolt.fine_tune(
            model_name=model_name,
            tokenizer_name=model_name,
            model_class="AutoModelForSequenceClassification",
            tokenizer_class="AutoTokenizer",
            num_train_epochs=1,
            per_device_batch_size=2,
            precision=precision,
            quantization=quantization,
            lora_config=lora_config,
            use_accelerate=use_accelerate,
            device_map="cuda:0",
            data_masked=False,
        )
        output_dir = commonsense_bolt.output.output_folder
        assert os.path.exists(
            os.path.join(commonsense_bolt.output.output_folder, "model", "pytorch_model.bin")
        ) or os.path.exists(os.path.join(commonsense_bolt.output.output_folder, "model", "adapter_model.bin"))
        assert os.path.exists(
            os.path.join(commonsense_bolt.output.output_folder, "model", "config.json")
        ) or os.path.exists(os.path.join(commonsense_bolt.output.output_folder, "model", "adapter_config.json"))
        assert os.path.exists(os.path.join(commonsense_bolt.output.output_folder, "model", "training_args.bin"))

        del commonsense_bolt.model
        del commonsense_bolt.tokenizer
        torch.cuda.empty_cache()

        try:
            os.remove(os.path.join(commonsense_bolt.output.output_folder, "model", "pytorch_model.bin"))
            os.remove(os.path.join(commonsense_bolt.output.output_folder, "model", "adapter_model.bin"))
            os.remove(os.path.join(commonsense_bolt.output.output_folder, "model", "config.json"))
            os.remove(os.path.join(commonsense_bolt.output.output_folder, "model", "adapter_config.json"))
            os.remove(os.path.join(commonsense_bolt.output.output_folder, "model", "training_args.bin"))
        except Exception as _:
            pass

    except Exception as e:
        del commonsense_bolt.model
        del commonsense_bolt.tokenizer
        torch.cuda.empty_cache()
        raise


# Test for computing metrics
def test_commonsense_bolt_compute_metrics(commonsense_bolt):
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    metrics = commonsense_bolt.compute_metrics(eval_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
