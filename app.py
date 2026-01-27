import os
import logging
import requests
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- APP CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

DEFAULT_MODEL = "phi"  

Settings.llm = Ollama(model=DEFAULT_MODEL, request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(model_name=DEFAULT_MODEL, request_timeout=600.0)

index = None

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global index
    try:
        if "file" not in request.files:
            return jsonify({"status": "No file uploaded"}), 400

        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        logger.info(f"Indexing file: {filepath}")

        documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
        index = VectorStoreIndex.from_documents(documents)

        return jsonify({"status": f"{file.filename} indexed successfully!"})

    except Exception as e:
        logger.error(e)
        return jsonify({"status": f"Error: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    global index

    if index is None:
        return jsonify({"response": "Please upload a document first."})

    data = request.json
    query = data.get("query")
    model = data.get("model", DEFAULT_MODEL)

    try:
        Settings.llm = Ollama(model=model, request_timeout=600.0)
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return jsonify({"response": str(response)})

    except Exception as e:
        logger.error(e)
        return jsonify({"response": "Model failed. Try smaller input or model."})

@app.route("/scrape", methods=["POST"])
def scrape():
    try:
        data = request.json
        url = data.get("url")
        model = data.get("model", DEFAULT_MODEL)

        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15)

        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 30]
        web_text = " ".join(paragraphs[:5])

        if not web_text:
            return jsonify({"summary": "No readable text found."})

        Settings.llm = Ollama(model=model, request_timeout=600.0)
        summary = Settings.llm.complete(
            f"Summarize the following in 2 sentences:\n\n{web_text}"
        )

        return jsonify({"summary": str(summary)})

    except Exception as e:
        logger.error(e)
        return jsonify({"summary": "Web scraping failed."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
