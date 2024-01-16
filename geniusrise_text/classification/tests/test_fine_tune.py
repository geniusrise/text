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

import pandas as pd
import pytest
import yaml  # type: ignore
from datasets import Dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyarrow import feather
from pyarrow import parquet as pq

from geniusrise_text import TextClassificationFineTuner

# Models to test
MODELS_TO_TEST = {
    # fmt: off
    "bert-base-uncased": ["LABEL_0", "LABEL_1", "LABEL_2"],
    "bert-large-uncased": ["LABEL_0", "LABEL_1", "LABEL_2"],
    "distilroberta-base": ["LABEL_0", "LABEL_1", "LABEL_2"],
    "xlm-roberta-large": ["LABEL_0", "LABEL_1", "LABEL_2"],
    "albert-base-v2": ["LABEL_0", "LABEL_1", "LABEL_2"],
    "cardiffnlp/twitter-roberta-base-2022-154m": ["LABEL_0", "LABEL_1", "LABEL_2"],
    "cardiffnlp/twitter-roberta-base": ["LABEL_0", "LABEL_1", "LABEL_2"],
    # fmt: on
}


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [{"text": f"text_{i}", "label": f"label_{i % 2}"} for i in range(10)]
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
            ET.SubElement(record, "text").text = item["text"]
            ET.SubElement(record, "label").text = item["label"]
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


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


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
    create_dataset_in_format(tmpdir + "/test", ext)
    return tmpdir, ext


@pytest.fixture
def classification_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()
    klass = TextClassificationFineTuner(
        input=input,
        output=output,
        state=state,
    )

    return klass


def test_classification_bolt_init(classification_bolt, model):
    model_name, labels = model
    tokenizer_name = model_name
    model_class = "AutoModelForSequenceClassification"
    tokenizer_class = "AutoTokenizer"

    classification_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    assert classification_bolt.model is not None
    assert classification_bolt.tokenizer is not None
    assert classification_bolt.input is not None
    assert classification_bolt.output is not None
    assert classification_bolt.state is not None


def test_load_dataset_all_formats(classification_bolt, dataset_file, model):
    tmpdir, ext = dataset_file
    dataset_path = os.path.join(tmpdir, "train")

    model_name, labels = model
    tokenizer_name = model_name
    model_class = "AutoModelForSequenceClassification"
    tokenizer_class = "AutoTokenizer"

    classification_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    dataset = classification_bolt.load_dataset(dataset_path)
    assert dataset is not None
    assert len(dataset) == 10


# Test for fine-tuning
def test_classification_bolt_fine_tune(classification_bolt, dataset_file, model):
    tmpdir, ext = dataset_file
    classification_bolt.input.input_folder = tmpdir

    model_name, labels = model
    tokenizer_name = model_name
    model_class = "AutoModelForSequenceClassification"
    tokenizer_class = "AutoTokenizer"
    # kwargs = {"model_"}

    classification_bolt.fine_tune(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
        num_train_epochs=2,
        per_device_batch_size=2,
        evaluate=True,
        precision="float16",
    )

    output_dir = classification_bolt.output.output_folder
    assert os.path.isfile(os.path.join(output_dir + "/model", "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(output_dir + "/model", "config.json"))
    assert os.path.isfile(os.path.join(output_dir + "/model", "training_args.bin"))
