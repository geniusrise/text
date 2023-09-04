# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import tempfile

import numpy as np
import pytest
from datasets import load_dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from transformers import EvalPrediction

from huggingface.base import HuggingFaceFineTuner


class TestHuggingFaceFineTuner(HuggingFaceFineTuner):
    def load_dataset(self, dataset_path, **kwargs):
        dataset = load_dataset("glue", "mrpc", split="train[:100]")
        dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=512,
            ),
            batched=True,
        ).map(lambda examples: {"labels": examples["label"]}, batched=True)
        print(dataset)
        return dataset


@pytest.fixture
def bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-🤗-output")
    state = InMemoryState()

    return TestHuggingFaceFineTuner(
        input=input,
        output=output,
        state=state,
        eval=False,
    )


def test_bolt_init(bolt):
    assert bolt.input is not None
    assert bolt.output is not None
    assert bolt.state is not None


def test_load_dataset(bolt):
    bolt.model_name = "bert-base-uncased"
    bolt.tokenizer_name = "bert-base-uncased"
    bolt.model_class = "BertForSequenceClassification"
    bolt.tokenizer_class = "BertTokenizer"
    bolt.load_models()
    dataset = bolt.load_dataset("fake_path")
    assert dataset is not None
    assert len(dataset) == 100


def test_fine_tune(bolt):
    bolt.fine_tune(
        model_name="bert-base-uncased",
        tokenizer_name="bert-base-uncased",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        model_class="BertForSequenceClassification",
        tokenizer_class="BertTokenizer",
        eval=False,
    )

    # Check that model files are created in the output directory
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))


def test_compute_metrics(bolt):
    # Mocking an EvalPrediction object
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = bolt.compute_metrics(eval_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_upload_to_hf_hub(bolt):
    bolt.fine_tune(
        model_name="bert-base-uncased",
        tokenizer_name="bert-base-uncased",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        model_class="BertForSequenceClassification",
        tokenizer_class="BertTokenizer",
        eval=False,
        hf_repo_id="ixaxaar/geniusrise-hf-base-test-repo",
        hf_commit_message="testing base fine tuner",
        hf_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
        hf_private=False,
        hf_create_pr=True,
    )

    assert True
