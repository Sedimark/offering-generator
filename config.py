import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    # Model configuration
    "model_name": "Qwen/Qwen2.5-3B",
    "checkpoint_path": MODEL_DIR / "checkpoint-40",  # PEFT checkpoint
    "max_length": 8192,

    # Generation parameters
    "generation": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 55,
        "repetition_penalty": 1.2,
        "num_beams": 4,
        "max_new_tokens": 8192,
        "early_stopping": True,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
    },

    # Server configuration
    "server": {
        "host": "0.0.0.0",
        "port": 8082,
        "workers": 1,
    },

    # Gradio configuration
    "gradio": {
        "port": 7860,
        "share": True,
    },

    # Output configuration
    "output_dir": OUTPUT_DIR,
    "schema_file": "schema_structure.json",  # Optional

    # JWT (for API security)
    "jwt_secret": os.getenv("JWT_SECRET", "SEDIMARK-deployment-key"),
}

# SEDIMARK context (standard namespaces)
SEDIMARK_CONTEXT = {
    "schema": "https://schema.org/",
    "ex": "http://example.org/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "dcterms": "http://purl.org/dc/terms/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dcat": "http://www.w3.org/ns/dcat#",
    "odrl": "http://www.w3.org/ns/odrl/2/",
    "prov": "http://www.w3.org/ns/prov#",
    "sedimark": "https://w3id.org/sedimark/ontology#",
    "dct": "http://purl.org/dc/terms/",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "@vocab": "https://w3id.org/sedimark/ontology#"
}
