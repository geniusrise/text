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
from typing import Any, Dict, Optional

import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger

from .bulk import TextBulk

# Define a global lock for sequential access control
sequential_lock = threading.Lock()


def sequential_tool():
    with sequential_lock:
        # Yield to signal that the request can proceed
        yield


# Register the custom tool
cherrypy.tools.sequential = cherrypy.Tool("before_handler", sequential_tool)


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
            model_name (str): The name of the pre-trained language model.
            model_class (str, optional): The name of the class of the pre-trained language model. Defaults to "AutoModelForCausalLM".
            tokenizer_class (str, optional): The name of the class of the tokenizer used to preprocess input text. Defaults to "AutoTokenizer".
            use_cuda (bool, optional): Whether to use a GPU for inference. Defaults to False.
            precision (str, optional): The precision to use for the pre-trained language model. Defaults to "float16".
            quantization (int, optional): The level of quantization to use for the pre-trained language model. Defaults to 0.
            device_map (str | Dict | None, optional): The mapping of devices to use for inference. Defaults to "auto".
            max_memory (Dict[int, str], optional): The maximum memory to use for inference. Defaults to {0: "24GB"}.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            awq_enabled (bool): Whether to use AWQ for model optimization. Default is False.
            flash_attention (bool): Whether to use flash attention 2. Default is False.
            endpoint (str, optional): The endpoint to listen on. Defaults to "*".
            port (int, optional): The port to listen on. Defaults to 3000.
            cors_domain (str, optional): The domain to allow CORS requests from. Defaults to "http://localhost:3000".
            username (Optional[str], optional): The username to use for authentication. Defaults to None.
            password (Optional[str], optional): The password to use for authentication. Defaults to None.
            **model_args (Any): Additional arguments to pass to the pre-trained language model.
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
        self.flash_attention = flash_attention
        self.awq_enabled = awq_enabled
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
                    "tools.auth_basic.on": True,
                    "tools.auth_basic.realm": "geniusrise",
                    "tools.auth_basic.checkpassword": self.validate_password,
                    "tools.CORS.on": True,
                }
            }
        else:
            # Configuration without authentication
            conf = {"/": {"tools.CORS.on": True}}

        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", conf)
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()


def error_page(status, message, traceback, version):
    response = {
        "status": status,
        "message": message,
    }
    return json.dumps(response)
