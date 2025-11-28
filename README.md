# SEDIMARK Offering Generator

A streamlined offering generation system for the SEDIMARK infrastructure. This toolbox runs on QWEN-7B model and utilizes PEFT fine-tuning to produce SEDIMARK-compliant JSON-LD offerings. The weights for the models are [here](https://example.com/will-upload-and-update-shortly) and are expected to be in the `models/` directory.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sedimark/offering-generator.git

# Clone or copy this directory
cd offering-generator

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup

Place your PEFT checkpoint in the `models/` directory. The expected structure would look something like this:

```
offering-generator/
└── models/
    └── checkpoint-40/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── tokenizer.json
```

Update `config.py` if your checkpoint path differs.

### 3. Run the API Server

```bash
python run_api.py
```

The API will be available at `http://localhost:8082`

### 4. Run the Gradio Interface

```bash
python run_gradio.py
```

The interface will be available at `http://localhost:7860`

## Project Structure

```
sedimark-clean/
├── api.py                 # FastAPI server
├── gradio_app.py         # Gradio interface
├── model_loader.py       # Model loading utilities
├── offering_generator.py # Core offering generation logic
├── config.py            # Configuration
├── run_api.py           # API runner script
├── run_gradio.py        # Gradio runner script
├── requirements.txt     # Dependencies
├── README.md           # This file
├── models/             # Model checkpoints (create this)
└── outputs/            # Generated offerings (auto-created)
```

## Configuration

Edit `config.py` to customize:

- Model paths and checkpoint locations
- Generation parameters (temperature, top_p, etc.)
- Server ports and settings
- Output directories

## API Endpoints

### Core Endpoints

- `GET /` - Service information
- `POST /generate` - Generate offering with detailed control
- `GET /api/offerings/generate` - Simple generation endpoint
- `GET /health` - Health check
- `GET /api/offerings/list` - List generated offerings

### Example API Usage

```bash
# Simple generation
curl "http://localhost:8082/api/offerings/generate?prompt=Generate%20IoT%20sensor%20offering"

# Advanced generation
curl -X POST "http://localhost:8082/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a SEDIMARK offering for smart city sensors",
    "temperature": 0.1,
    "max_new_tokens": 4096,
    "use_schema": false
  }'
```

## Usage Examples

### Generate IoT Sensor Offering

```python
from offering_generator import SEDIMARKOfferingGenerator
from model_loader import ModelLoader

# Initialize
loader = ModelLoader()
loader.load_model()
generator = SEDIMARKOfferingGenerator(loader)

# Generate
offering = generator.generate_offering(
    prompt="Generate offering for temperature sensors in Tokyo",
    use_context=False,
    temperature=0.1
)

print(json.dumps(offering, indent=2))
```

### Batch Generation

```bash
# Generate multiple offerings
curl "http://localhost:8082/api/offerings/generate?count=3&prompt=Smart%20city%20data"
```

This clean implementation provides the same powerful SEDIMARK generation capabilities with dramatically reduced complexity, making it ideal for production deployment.
