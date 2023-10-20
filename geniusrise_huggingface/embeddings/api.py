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

from typing import Any, Dict, Optional

import cherrypy
from sentence_transformers import SentenceTransformer

from geniusrise_huggingface.base import HuggingFaceAPI
from geniusrise_huggingface.embeddings.embeddings import (
    generate_sentence_transformer_embeddings,
    generate_embeddings,
    generate_contiguous_embeddings,
    generate_combination_embeddings,
    generate_permutation_embeddings,
)


class EmbeddingsAPI(HuggingFaceAPI):
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def sbert_embeddings(self, **kwargs: Any) -> Dict[str, Any]:
        data = cherrypy.request.json
        sentences = data.get("sentences")
        batch_size = data.get("batch_size", 32)

        embeddings = generate_sentence_transformer_embeddings(
            sentences=sentences, model=self.sentence_transformer_model, use_cuda=self.use_cuda, batch_size=batch_size
        )
        return {"embeddings": embeddings.tolist()}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def embeddings(self, **kwargs: Any) -> Dict[str, Any]:
        data = cherrypy.request.json
        term = data.get("term")

        embeddings = generate_embeddings(
            term=term,
            model=self.model,
            tokenizer=self.tokenizer,
            output_key="last_hidden_state",
            use_cuda=self.use_cuda,
        )
        return {"embeddings": embeddings.tolist()}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def embeddings_contiguous(self, **kwargs: Any) -> Dict[str, Any]:
        data = cherrypy.request.json
        sentence = data.get("sentence")

        embeddings = generate_contiguous_embeddings(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tokenizer,
            output_key="last_hidden_state",
            use_cuda=self.use_cuda,
        )
        return {"embeddings": embeddings}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def embeddings_combinations(self, **kwargs: Any) -> Dict[str, Any]:
        data = cherrypy.request.json
        sentence = data.get("sentence")

        embeddings = generate_combination_embeddings(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tokenizer,
            output_key="last_hidden_state",
            use_cuda=self.use_cuda,
        )
        return {"embeddings": embeddings}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def embeddings_permutations(self, **kwargs: Any) -> Dict[str, Any]:
        data = cherrypy.request.json
        sentence = data.get("sentence")

        embeddings = generate_permutation_embeddings(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tokenizer,
            output_key="last_hidden_state",
            use_cuda=self.use_cuda,
        )
        return {"embeddings": embeddings}

    def listen(  # type: ignore
        self,
        model_name: str,
        model_class_name: str = "AutoModelForCausalLM",
        tokenizer_class_name: str = "AutoTokenizer",
        sentence_transformer_model: str = "paraphrase-MiniLM-L6-v2",
        use_cuda: bool = False,
        precision: str = "float16",
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
        endpoint: str = "*",
        port: int = 3000,
        cors_domain: str = "http://localhost:3000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **model_args: Any,
    ) -> None:
        self.model_name = model_name
        self.model_class_name = model_class_name
        self.tokenizer_class_name = tokenizer_class_name
        self.use_cuda = use_cuda
        self.precision = precision
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.model_args = model_args

        self.model, self.tokenizer = self.load_huggingface_model(
            model_name=self.model_name,
            model_class_name=self.model_class_name,
            tokenizer_class_name=self.tokenizer_class_name,
            use_cuda=self.use_cuda,
            precision=self.precision,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            **self.model_args,
        )
        self.sentence_transformer_model = SentenceTransformer(model_name, device="cuda" if use_cuda else "cpu")

        def CORS():
            cherrypy.response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
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
            }
        )

        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", {"/": {"tools.CORS.on": True}})
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()
