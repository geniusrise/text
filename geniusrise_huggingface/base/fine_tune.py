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

import logging
import os
from abc import abstractmethod
from typing import Dict, Optional, List

import numpy as np
from datasets import Dataset, DatasetDict
from geniusrise import BatchInput, BatchOutput, Bolt, State
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoConfig, EvalPrediction, Trainer, TrainingArguments, AutoModel
from accelerate import infer_auto_device_map, init_empty_weights
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead
import torch


class HuggingFaceFineTuner(Bolt):
    """
    A bolt for fine-tuning Hugging Face models.

    This bolt uses the Hugging Face Transformers library to fine-tune a pre-trained model.
    It uses the `Trainer` class from the Transformers library to handle the training.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        """
        Initialize the bolt.

        Args:
            input (BatchInput): The batch input data.
            output (OutputConfig): The output data.
            state (State): The state manager.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.input = input
        self.output = output
        self.state = state

        self.model_name: Optional[str] = None
        self.tokenizer_name: Optional[str] = None
        self.model_class: Optional[str] = None
        self.tokenizer_class: Optional[str] = None
        self.eval: bool = False
        self.device = "cuda"

        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None

        self.log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_dataset(self, dataset_path: str, **kwargs) -> Dataset | DatasetDict | Optional[Dataset]:
        """
        Load a dataset from a file.

        Args:
            dataset_path (str): The path to the dataset file.
            **kwargs: Additional keyword arguments to pass to the `load_dataset` method.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def preprocess_data(self, **kwargs):
        """Load and preprocess the dataset"""
        try:
            self.input.copy_from_remote()
            train_dataset_path = os.path.join(self.input.get(), "train")
            eval_dataset_path = os.path.join(self.input.get(), "eval")
            self.train_dataset = self.load_dataset(train_dataset_path, **kwargs)
            if self.eval:
                self.eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
        except Exception as e:
            self.log.exception(f"Failed to preprocess data: {e}")
            raise

    def load_models(
        self,
        model_name: str,
        tokenizer_name: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        model_class: str = "AutoModel",
        tokenizer_class: str = "AutoTokenizer",
        device_map: str | dict = "auto",
        precision: str = "bfloat16",
        quantization: Optional[int] = None,
        lora_config: Optional[dict] = None,
        use_accelerate: bool = False,
        accelerate_no_split_module_classes: List[str] = [],
    ):
        """Load the model and tokenizer"""
        try:
            # Determine the torch dtype based on precision
            if precision == "float16":
                torch_dtype = torch.float16
            elif precision == "float32":
                torch_dtype = torch.float32
            elif precision == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                raise ValueError("Unsupported precision. Choose from 'float32', 'float16', 'bfloat16'.")

            if use_accelerate:
                with init_empty_weights():
                    model = AutoModel.from_pretrained(
                        model_name,
                        peft_config=lora_config,
                        device_map=device_map,
                    )
                    if device_map == "float16":
                        device_map = infer_auto_device_map(
                            model,
                            dtype="float16",
                            accelerate_no_split_module_classes=accelerate_no_split_module_classes,
                        )
                    elif device_map == "bfloat16":
                        device_map = infer_auto_device_map(
                            model,
                            dtype="bfloat16",
                            accelerate_no_split_module_classes=accelerate_no_split_module_classes,
                        )
                    else:
                        device_map = infer_auto_device_map(
                            model,
                            accelerate_no_split_module_classes=accelerate_no_split_module_classes,
                        )

            # Load model and tokenizer
            # If peft_config is provided, apply PEFT optimizations
            if lora_config and quantization and quantization > 8:
                lora_conf = LoraConfig(**lora_config)
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    peft_config=lora_config,
                    device_map=device_map,
                )
            #  if PEFT config is provided and
            elif lora_config and quantization == 8:
                lora_conf = LoraConfig(**lora_config)
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    peft_config=lora_config,
                    device_map=device_map,
                )
            elif lora_config and quantization == 4:
                lora_conf = LoraConfig(**lora_config)
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    load_in_4bit=True,
                    peft_config=lora_config,
                    device_map=device_map,
                )
            else:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.config = AutoConfig.from_pretrained(os.path.join(self.input.get(), "/model"))
                    self.model = getattr(__import__("transformers"), str(self.model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        config=self.config,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                    )
                else:
                    self.config = AutoConfig.from_pretrained(self.model_name)
                    self.model = getattr(__import__("transformers"), str(self.model_class)).from_pretrained(
                        self.model_name,
                        config=self.config,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                    )

                if self.tokenizer_name.lower() == "local":  # type: ignore
                    self.tokenizer = getattr(__import__("transformers"), str(self.tokenizer_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model")
                    )
                else:
                    self.tokenizer = getattr(__import__("transformers"), str(self.tokenizer_class)).from_pretrained(
                        self.tokenizer_name
                    )
        except Exception as e:
            self.log.exception(f"Failed to load model: {e}")
            raise

    def upload_to_hf_hub(self):
        """Upload the model and tokenizer to Hugging Face Hub."""
        try:
            if self.model:
                self.model.push_to_hub(
                    repo_id=self.hf_repo_id,
                    commit_message=self.hf_commit_message,
                    token=self.hf_token,
                    private=self.hf_private,
                    create_pr=self.hf_create_pr,
                )
            if self.tokenizer:
                self.tokenizer.push_to_hub(
                    repo_id=self.hf_repo_id,
                    commit_message=self.hf_commit_message,
                    token=self.hf_token,
                    private=self.hf_private,
                    create_pr=self.hf_create_pr,
                )
        except Exception as e:
            self.log.exception(f"Failed to upload model to huggingface hub: {e}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Optional[Dict[str, float]] | Dict[str, float]:
        """
        Compute metrics for evaluation. This class implements a simple classification evaluation, tasks should ideally override this.

        Args:
            eval_pred (EvalPrediction): The evaluation predictions.

        Returns:
            dict: The computed metrics.
        """
        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        labels = labels[0] if isinstance(labels, tuple) else labels
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_recall_fscore_support(labels, predictions, average="binary")[0],
            "recall": precision_recall_fscore_support(labels, predictions, average="binary")[1],
            "f1": precision_recall_fscore_support(labels, predictions, average="binary")[2],
        }

    def fine_tune(
        self,
        model_name: str,
        tokenizer_name: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        model_class: str = "AutoModel",
        tokenizer_class: str = "AutoTokenizer",
        device_map: str | dict = "auto",
        device: str = "cuda",
        precision: str = "bfloat16",
        quantization: Optional[int] = None,
        lora_config: Optional[dict] = None,
        use_accelerate: bool = False,
        accelerate_no_split_module_classes: List[str] = [],
        eval: bool = False,
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: bool = True,
        hf_create_pr: bool = False,
        **kwargs,
    ):
        """
        Fine-tune the model.

        Args:
            model_name (str): The pre-trained model name.
            tokenizer_name (str): The pre-trained tokenizer name.
            num_train_epochs (int): Total number of training epochs to perform.
            per_device_train_batch_size (int): Batch size per device during training.
            model_class (str, optional): The model class to use. Defaults to "AutoModel".
            tokenizer_class (str, optional): The tokenizer class to use. Defaults to "AutoTokenizer".
            eval (bool, optional): Whether to evaluate the model after training. Defaults to False.
            hf_repo_id (str, optional): The Hugging Face repo ID. Defaults to None.
            hf_commit_message (str, optional): The Hugging Face commit message. Defaults to None.
            hf_token (str, optional): The Hugging Face token. Defaults to None.
            hf_private (bool, optional): Whether to make the repo private. Defaults to True.
            hf_create_pr (bool, optional): Whether to create a pull request. Defaults to False.
            lora_config (dict, optional): Configuration for PEFT LoRA optimization. Defaults to None.
            use_accelerate (bool, optional): Whether to use accelerate for distributed training. Defaults to False.
            **kwargs: Additional keyword arguments for training.

        Raises:
            Exception: If any step in the fine-tuning process fails.
        """
        try:
            # Save everything
            self.model_name = model_name
            self.tokenizer_name = tokenizer_name
            self.output_dir = self.output.output_folder
            self.num_train_epochs = num_train_epochs
            self.per_device_train_batch_size = per_device_train_batch_size
            self.model_class = model_class
            self.tokenizer_class = tokenizer_class
            self.eval = eval
            self.hf_repo_id = hf_repo_id
            self.hf_commit_message = hf_commit_message
            self.hf_token = hf_token
            self.hf_private = hf_private
            self.hf_create_pr = hf_create_pr
            self.device = device

            self.load_models(
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                model_class=model_class,
                tokenizer_class=tokenizer_class,
                device_map=device_map,
                precision=precision,
                quantization=quantization,
                lora_config=lora_config,
                use_accelerate=use_accelerate,
                accelerate_no_split_module_classes=accelerate_no_split_module_classes,
            )

            # Load dataset
            dataset_kwargs = {k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k}
            self.preprocess_data(**dataset_kwargs)

            # Separate training and evaluation arguments
            trainer_kwargs = {k.replace("trainer_", ""): v for k, v in kwargs.items() if "trainer_" in k}
            training_kwargs = {k.replace("training_", ""): v for k, v in kwargs.items() if "training_" in k}

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output_dir, "model"),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                **training_kwargs,
            )

            if self.device and self.model:
                self.model.to(device)

            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset if self.eval else None,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                data_collator=self.data_collator if hasattr(self, "data_collator") else None,
                **trainer_kwargs,
            )

            # Train the model
            trainer.train()
            trainer.save_model()

            if self.eval:
                eval_result = trainer.evaluate()
                self.log.info(f"Evaluation results: {eval_result}")

            # Save the model configuration to Hugging Face Hub if hf_repo_id is not None
            if self.hf_repo_id:
                self.config.save_pretrained(os.path.join(self.output_dir, "model"))
                self.upload_to_hf_hub()
        except Exception as e:
            self.log.exception(f"Failed to fine tune model: {e}")
            self.state.set_state(self.id, {"success": False, "exception": str(e)})
            raise
        self.state.set_state(self.id, {"success": True})
