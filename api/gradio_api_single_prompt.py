#!/usr/bin/env python3
"""
SEDIMARK LLM Gradio Interface - Updated to match API changes
Matches the behavior from api_sedimark.py
"""

import os
import sys
import logging
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import re

# Add the config directory to Python path
config_dir = "."
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

# Import from the correct location matching API
try:
    from src.models.evaluator import JSONLLMEvaluator
    from config.config import CONFIG
    print("Successfully imported JSONLLMEvaluator and CONFIG")
except ImportError as e:
    print(f"Import error: {e}")
    sys.path.insert(0, os.path.join(config_dir, "src"))
    from models.evaluator import JSONLLMEvaluator
    from config.config import CONFIG

# Import utilities with fallback
try:
    from src.utils.json_utils import validate_json_output
except ImportError:
    def validate_json_output(output_str):
        """Basic JSON validation fallback"""
        try:
            json.loads(output_str)
            return output_str, True
        except json.JSONDecodeError:
            if output_str.strip().endswith(','):
                output_str = output_str.strip()[:-1]
            try:
                json.loads(output_str)
                return output_str, True
            except:
                return output_str, False

try:
    from src.utils.memory import deep_cleanup
except ImportError:
    def deep_cleanup():
        """Basic memory cleanup fallback"""
        torch.cuda.empty_cache()
        import gc
        gc.collect()

print(f"Loaded config from: {config_dir}/config/config.py")
print(f"Base model path: {CONFIG['model_name']}")
print(f"Cache directory: {CONFIG['cache_dir']}")

# Configure for PEFT/LoRA loading
CONFIG["quantization"]["inference"]["enabled"] = False
CONFIG["quantization"]["inference"]["load_in_4bit"] = False
CONFIG["quantization"]["inference"]["bnb_4bit_use_double_quant"] = False
CONFIG["qlora"]["enabled"] = True

# Checkpoint path - updated to match API
CHECKPOINT_PATH = "Model_weights/Exp_3B/checkpoints/checkpoint-40"

print(f"Using checkpoint path: {CHECKPOINT_PATH}")

# Initialize logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global variables
evaluator = None
offering_generator = None

# Output directory
OUTPUT_DIR = "./Gradio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)



class SEDIMARKOfferingGenerator:
    """SEDIMARK offering generator matching API implementation"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.schema_data = None
        self.load_schema()
        
    def load_schema(self):
        """Load schema file if available"""
        try:
            schema_file = CONFIG.get("schema_file")
            if schema_file and os.path.exists(schema_file):
                with open(schema_file, 'r') as f:
                    self.schema_data = json.load(f)
                logger.info(f"Loaded schema with {len(self.schema_data.get('@graph', []))} class definitions")
        except Exception as e:
            logger.warning(f"Could not load schema: {e}")
            self.schema_data = None
    
    def get_default_context(self) -> Dict[str, str]:
        """Get the standard SEDIMARK @context"""
        return {
            "schema": "https://schema.org/",
            "ex": "http://example.org/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "dcterms": "http://purl.org/dc/terms/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dcat": "http://www.w3.org/ns/dcat#",
            "odrl": "http://www.w3.org/ns/odrl/2/",
            "sdm-vocab": "https://w3id.org/sedimark/vocab/",
            "prov": "http://www.w3.org/ns/prov#",
            "sedimark": "https://w3id.org/sedimark/ontology#",
            "sedi": "https://w3id.org/sedimark/ontology#",
            "dct": "http://purl.org/dc/terms/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "@vocab": "https://w3id.org/sedimark/ontology#"
        }
    
    def extract_json_from_output(self, raw_output: str) -> str:
        """Extract JSON from model output"""
        
        # Remove prompt if echoed
        if "Begin JSON output now:" in raw_output:
            raw_output = raw_output.split("Begin JSON output now:")[-1].strip()
        
        if "JSON output:" in raw_output:
            raw_output = raw_output.split("JSON output:")[-1].strip()
        
        if "<|json_output|>" in raw_output:
            raw_output = raw_output.split("<|json_output|>")[-1].strip()
        
        # Remove markdown code blocks
        if "```json" in raw_output:
            match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in raw_output:
            match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Try to find JSON object starting with { or [
        # Look for the first { or [ and find its matching closing bracket
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = raw_output.find(start_char)
            if start_idx != -1:
                # Use a simple bracket counter
                count = 0
                in_string = False
                escape_next = False
                
                for i in range(start_idx, len(raw_output)):
                    char = raw_output[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                        
                    if char == '\\':
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        
                    if not in_string:
                        if char == start_char:
                            count += 1
                        elif char == end_char:
                            count -= 1
                            if count == 0:
                                json_str = raw_output[start_idx:i+1]
                                # Clean up common issues
                                json_str = re.sub(r',\s*}', '}', json_str)
                                json_str = re.sub(r',\s*]', ']', json_str)
                                return json_str
        
        # If no valid JSON structure found, return cleaned output
        return raw_output.strip()

    def generate_offering(self, 
                     prompt: str = None,
                     use_context: bool = False,
                     context: Dict[str, Any] = None,
                     use_schema: bool = False,
                     max_new_tokens: int = 8192,
                     temperature: float = 0.1,
                     top_p: float = 0.95,
                     num_beams: int = 4) -> Dict[str, Any]:
        """Generate a SEDIMARK-compliant offering matching API behavior"""
        
        input_text = prompt or "Generate a SEDIMARK offering for IoT sensor data"
        
        # Fixed prompt with proper JSON structure
        formatted_prompt = f"""Generate a valid JSON-LD SEDIMARK offering based on this description:
    {input_text}

    Instructions: Output ONLY valid JSON. No explanatory text before or after.
    The JSON must have this exact structure:
    {{
    "@graph": [
        {{
        "@id": "ex:offering_xyz",
        "@type": "sedimark:Offering",
        "dcterms:title": "...",
        "dcterms:description": "...",
        "sedimark:hasAsset": [{{"@id": "ex:asset_xyz"}}]
        }},
        {{
        "@id": "ex:asset_xyz",
        "@type": "sedimark:Asset",
        "dcterms:title": "...",
        "dcterms:description": "..."
        }}
    ],
    "@context": {{
        "sedimark": "https://w3id.org/sedimark/ontology#",
        "dcterms": "http://purl.org/dc/terms/",
        "ex": "http://example.org/"
    }}
    }}

    Begin JSON output now:
    """
        
        # Tokenize
        inputs = self.evaluator.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.evaluator.max_length
        ).to(self.evaluator.device)
        
        # Generation parameters
        generation_params = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': 55,
            'repetition_penalty': 1.2,
            'num_beams': num_beams,
            'early_stopping': True,
            'pad_token_id': self.evaluator.tokenizer.pad_token_id,
            'eos_token_id': self.evaluator.tokenizer.eos_token_id,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
        }
        
        with torch.inference_mode():
            try:
                deep_cleanup()
                
                precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast('cuda', dtype=precision_dtype):
                    outputs = self.evaluator.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **generation_params
                    )
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM error, reducing generation parameters...")
                    deep_cleanup()
                    
                    generation_params.update({
                        'num_beams': 1,
                        'max_new_tokens': 4096,
                        'do_sample': False,
                    })
                    
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = self.evaluator.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            **generation_params
                        )
                else:
                    raise
        
        # Decode output
        raw_output = self.evaluator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from generated text
        json_text = self.extract_json_from_output(raw_output)
        
        # Try to parse as JSON
        try:
            offering = json.loads(json_text)
            
            # Validate structure
            if not isinstance(offering, dict):
                raise ValueError("Output is not a JSON object")
            
            # Add context if missing
            if "@context" not in offering:
                offering["@context"] = self.get_default_context()
            
            # Add graph if missing but has other content
            if "@graph" not in offering and len(offering) > 1:
                # Wrap existing content in @graph
                graph_content = {k: v for k, v in offering.items() if k != "@context"}
                offering = {
                    "@graph": [graph_content],
                    "@context": offering.get("@context", self.get_default_context())
                }
            
            return offering
            
        except (json.JSONDecodeError, ValueError) as e:
            # Return raw output with error
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw output: {raw_output[:500]}...")  # Log first 500 chars
            logger.debug(f"Extracted text: {json_text[:500]}...")
            
            return {
                "raw_output": json_text,
                "error": f"Invalid JSON generated: {str(e)}",
                "full_output": raw_output  # Include full output for debugging
            }
    
    def generate_fallback_offering(self) -> Dict[str, Any]:
        """Generate a fallback valid offering if generation fails"""
        offering_id = f"ex:offering_{uuid.uuid4().hex[:8]}"
        asset_id = f"ex:asset_{uuid.uuid4().hex[:8]}"
        
        return {
            "@graph": [
                {
                    "@id": offering_id,
                    "@type": "sedimark:Offering",
                    "dcterms:title": "Generated IoT Sensor Data Offering",
                    "dcterms:description": "Offering for IoT sensor data",
                    "dcterms:license": "CC-BY-4.0",
                    "dcat:themeTaxonomy": {"@id": "sdm-vocab:sdm"},
                    "sedimark:hasAsset": [{"@id": asset_id}]
                },
                {
                    "@id": asset_id,
                    "@type": "sedimark:Asset",
                    "dcterms:title": "IoT Sensor Dataset",
                    "dcterms:description": "Dataset from IoT sensors",
                    "dcterms:identifier": f"asset-{uuid.uuid4()}",
                    "dcterms:issued": {
                        "@value": datetime.now().isoformat() + "Z",
                        "@type": "xsd:dateTime"
                    },
                    "dcterms:creator": "SEDIMARK Generator",
                    "sedimark:offeredBy": {"@id": offering_id}
                }
            ],
            "@context": self.get_default_context()
        }

def verify_peft_checkpoint(checkpoint_path):
    """Verify PEFT checkpoint has required files"""
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors", 
        "tokenizer.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(checkpoint_path, file)):
            missing_files.append(file)
    
    if missing_files:
        raise RuntimeError(f"Missing PEFT files: {missing_files}")
    
    print(f"PEFT checkpoint verified: {checkpoint_path}")
    return True

def startup_model_loading():
    global evaluator, offering_generator
    
    try:
        logger.info("Starting model loading...")
        logger.info(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
        
        # Verify PEFT checkpoint
        verify_peft_checkpoint(CHECKPOINT_PATH)
        
        # Log adapter info
        try:
            with open(os.path.join(CHECKPOINT_PATH, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
                logger.info(f"LoRA config: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
                logger.info(f"Target modules: {adapter_config.get('target_modules')}")
        except Exception as e:
            logger.warning(f"Could not read adapter config: {e}")
        
        CONFIG["qlora"]["enabled"] = True
        logger.info("Loading PEFT/LoRA checkpoint")
        
        # Load evaluator matching API logic
        try:
            if hasattr(JSONLLMEvaluator, 'from_checkpoint_with_peft'):
                evaluator = JSONLLMEvaluator.from_checkpoint_with_peft(CHECKPOINT_PATH)
            elif hasattr(JSONLLMEvaluator, 'from_checkpoint'):
                evaluator = JSONLLMEvaluator.from_checkpoint(CHECKPOINT_PATH)
            else:
                logger.warning("Using standard initialization")
                evaluator = JSONLLMEvaluator()
                from peft import PeftModel
                logger.info("Loading PEFT adapter...")
                evaluator.model = PeftModel.from_pretrained(evaluator.model, CHECKPOINT_PATH)
                logger.info("PEFT adapter loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load with PEFT method: {e}")
            logger.info("Attempting fallback approach...")
            
            evaluator = JSONLLMEvaluator()
            from peft import PeftModel
            logger.info("Loading PEFT adapter...")
            evaluator.model = PeftModel.from_pretrained(evaluator.model, CHECKPOINT_PATH)
            logger.info("PEFT adapter loaded successfully")
        
        if evaluator is None or evaluator.model is None:
            raise RuntimeError("Failed to load model from checkpoint")
        
        # Initialize SEDIMARK offering generator
        offering_generator = SEDIMARKOfferingGenerator(evaluator)
        
        # Log model info
        vocab_size = len(evaluator.tokenizer)
        param_count = sum(p.numel() for p in evaluator.model.parameters())
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Model info:")
        logger.info(f"   - Vocabulary size: {vocab_size:,}")
        logger.info(f"   - Parameters: {param_count:,}")
        logger.info(f"   - Device: {evaluator.device}")
        logger.info(f"   - Max length: {evaluator.max_length}")
        
        return True
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def save_offering(offering_json: str, filename: str = ""):
    """Save offering to file"""
    try:
        if not offering_json.strip():
            return " No offering to save"
        
        # Generate filename if not provided
        if not filename.strip():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"offering_{timestamp}.json"
        elif not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save with UTF-8 encoding
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(offering_json)
        
        file_size = os.path.getsize(filepath)
        return f" Saved successfully!\n Path: {filepath}\n Size: {file_size} bytes"
        
    except Exception as e:
        return f" Save failed: {str(e)}"

def create_gradio_interface():
    """Create professional Gradio interface with SEDIMARK offering generation"""
    
    try:
        import gradio as gr
        print(" Gradio imported successfully")
    except ImportError:
        print(" Gradio not installed. Install with: pip install gradio")
        return None
    
    # Get current model status
    model_loaded = evaluator is not None and offering_generator is not None
    
    # Professional CSS styling
    css = """
    .gradio-container {
        max-width: 1600px !important;
        margin: 0 auto !important;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0;
    }
    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
    }
    .prompt-section {
        background: #f0f9ff;
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.1);
    }
    .params-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .generate-btn:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    .utility-btn {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        color: #475569 !important;
        font-weight: 500 !important;
        margin: 0.25rem !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    .utility-btn:hover {
        background: #e2e8f0 !important;
        border-color: #94a3b8 !important;
    }
    .metric-display {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    .prompt-tips {
        background: #fefce8;
        border: 1px solid #fde047;
        border-radius: 6px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.875rem;
        color: #713f12;
    }
    .generation-mode {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(
        title="SEDIMARK Offering Generator",
        theme=gr.themes.Default(),
        css=css
    ) as interface:
        
        # Header
        with gr.Row(elem_classes=["main-header"]):
            gr.HTML("""
            <div style="text-align: center;">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;"> SEDIMARK Offering Generator</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Generate SEDIMARK-compliant JSON-LD offerings</p>
            </div>
            """)
        
        # Model status
        with gr.Row():
            with gr.Column():
                status_html = f"""
                <div class="status-card">
                    <h3 style="margin-top: 0; color: #1e293b;"> System Status</h3>
                    <p><strong>Model:</strong> {' Ready' if model_loaded else ' Failed to Load'}</p>
                    <p><strong>Checkpoint:</strong> {os.path.basename(CHECKPOINT_PATH)}</p>
                    <p><strong>Type:</strong> PEFT/LoRA Fine-tuned (SEDIMARK-Optimized)</p>
                    <p><strong>Output Directory:</strong> {OUTPUT_DIR}</p>
                    {"<p><strong>Schema:</strong> " + (" Loaded" if offering_generator and offering_generator.schema_data else "⚠️ Not Available") + "</p>" if model_loaded else ""}
                </div>
                """
                gr.HTML(status_html)
        
        if model_loaded:
            # Generation Mode Selection
            with gr.Row():
                with gr.Column(elem_classes=["generation-mode"]):
                    gr.HTML("<h3 style='margin-top: 0; color: #1e40af;'> Generation Mode</h3>")
                    generation_mode = gr.Radio(
                        choices=[
                            "Standard (No Context)",
                            "With Context",
                            "Schema-Guided"
                        ],
                        value="Standard (No Context)",
                        label="Select Generation Mode",
                        info="Choose how the offering should be generated"
                    )
                    
                    mode_description = gr.HTML("""
                    <div style='margin-top: 1rem; padding: 0.75rem; background: #dbeafe; border-radius: 6px;'>
                        <strong>Standard:</strong> Generate offering without external context using embedded knowledge<br>
                        <strong>With Context:</strong> Include custom @context for namespaces<br>
                        <strong>Schema-Guided:</strong> Follow strict schema definitions (if available)
                    </div>
                    """)
            
            # Prompt Input Section
            with gr.Row():
                with gr.Column(elem_classes=["prompt-section"]):
                    gr.HTML("<h3 style='margin-top: 0; color: #0369a1;'> Offering Description</h3>")
                    
                    prompt_input = gr.Textbox(
                        label="Describe the offering you want to generate:",
                        lines=6,
                        placeholder="""Example prompts:
• Generate a SEDIMARK offering for smart-lock entry logs from residential doors in Tokyo
• Create an offering for temperature and humidity sensor data from industrial facilities
• Generate an offering for vehicle telemetry data including GPS, speed, and fuel consumption
• Create a data offering for energy consumption metrics from smart meters""",
                        value="",
                        interactive=True,
                        show_copy_button=True
                    )
                    
                    # Quick action buttons
                    with gr.Row():
                        clear_btn = gr.Button(" Clear", size="sm", elem_classes=["utility-btn"])
                        example_btn = gr.Button(" Load Example", size="sm", elem_classes=["utility-btn"])
                        with gr.Column():
                            char_count = gr.HTML("<small style='color: #64748b; float: right;'>Characters: 0</small>")
                    
                    # Context input (conditional)
                    context_input = gr.Textbox(
                        label="Custom @context (JSON format):",
                        lines=4,
                        placeholder='{"sedimark": "https://w3id.org/sedimark/ontology#", ...}',
                        visible=False,
                        interactive=True
                    )
                    
                    # Tips
                    gr.HTML("""
                    <div class="prompt-tips">
                        <strong> Tips for Better Offerings:</strong><br>
                        • Specify the data domain (IoT, energy, transportation, etc.)<br>
                        • Include temporal context (collection period, update frequency)<br>
                        • Mention geographical location or coverage area<br>
                        • Describe data attributes and quality metrics<br>
                        • Include licensing or access requirements
                    </div>
                    """)
            
            # Parameters section
            with gr.Row():
                with gr.Column(elem_classes=["params-section"]):
                    gr.HTML("<h3 style='margin-top: 0; color: #1e293b;'>⚙️ Generation Parameters</h3>")
                    
                    with gr.Row():
                        with gr.Column():
                            max_tokens_slider = gr.Slider(
                                minimum=512, 
                                maximum=8192, 
                                value=4096, 
                                step=256,
                                label="Max Tokens",
                                interactive=True
                            )
                        with gr.Column():
                            temp_slider = gr.Slider(
                                minimum=0.0, 
                                maximum=1.0, 
                                value=0.1, 
                                step=0.05,
                                label="Temperature",
                                interactive=True
                            )
                        with gr.Column():
                            top_p_slider = gr.Slider(
                                minimum=0.1, 
                                maximum=1.0, 
                                value=0.95, 
                                step=0.05,
                                label="Top-p",
                                interactive=True
                            )
                        with gr.Column():
                            num_beams_slider = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Beam Search",
                                interactive=True
                            )
            
            # Generate button
            with gr.Row():
                with gr.Column():
                    generate_button = gr.Button(
                        " Generate SEDIMARK Offering",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"]
                    )
            
            # Performance metrics
            with gr.Row():
                with gr.Column(elem_classes=["metric-display"]):
                    tokens_per_sec = gr.HTML("""
                    <div class="metric-value">-</div>
                    <div class="metric-label">Tokens/Second</div>
                    """)
                with gr.Column(elem_classes=["metric-display"]):
                    total_tokens = gr.HTML("""
                    <div class="metric-value">-</div>
                    <div class="metric-label">Output Tokens</div>
                    """)
                with gr.Column(elem_classes=["metric-display"]):
                    generation_time = gr.HTML("""
                    <div class="metric-value">-</div>
                    <div class="metric-label">Generation Time</div>
                    """)
                with gr.Column(elem_classes=["metric-display"]):
                    json_valid = gr.HTML("""
                    <div class="metric-value">-</div>
                    <div class="metric-label">JSON Status</div>
                    """)
            
            # Output section with tabs
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem(" Generated Offering"):
                        offering_output = gr.Code(
                            label="",
                            language="json",
                            lines=20,
                            interactive=False
                        )
                    
                    with gr.TabItem(" Validation"):
                        validation_output = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            elem_classes=["status-card"]
                        )
                    
                    with gr.TabItem(" Statistics"):
                        stats_output = gr.Textbox(
                            label="",
                            lines=8,
                            interactive=False,
                            elem_classes=["status-card"]
                        )
            
            # Save section
            with gr.Row():
                with gr.Column(scale=4):
                    save_filename = gr.Textbox(
                        label="Filename (optional)",
                        placeholder="offering_YYYYMMDD_HHMMSS.json",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    save_button = gr.Button(" Save", variant="secondary")
            
            save_status = gr.Textbox(
                label="Save Status",
                interactive=False,
                lines=2
            )
            
            # Event handler functions
            def update_mode_visibility(mode):
                """Update visibility based on generation mode"""
                return gr.update(visible=(mode == "With Context"))
            
            def clear_prompt():
                return ""
            
            def load_example():
                return """Generate a comprehensive SEDIMARK offering for smart city IoT sensor data including:
- Temperature, humidity, and air quality measurements
- Data from 500 sensors across Tokyo metropolitan area
- Collection period: January 2024 to December 2024
- Update frequency: Every 15 minutes
- Include quality metrics and data provision details
- Licensed under CC-BY-4.0"""
            
            def count_characters(text):
                char_count = len(text) if text else 0
                return f"<small style='color: #64748b; float: right;'>Characters: {char_count}</small>"
            
            def generate_offering_wrapper(prompt, mode, context_json, max_tokens, temp, top_p, num_beams):
                """Wrapper function for offering generation"""
                
                if not prompt.strip():
                    return (" Please enter a prompt", "", "", 
                           "<div class='metric-value'>-</div><div class='metric-label'>Tokens/Second</div>",
                           "<div class='metric-value'>-</div><div class='metric-label'>Output Tokens</div>",
                           "<div class='metric-value'>-</div><div class='metric-label'>Generation Time</div>",
                           "<div class='metric-value'>error</div><div class='metric-label'>JSON Status</div>")
                
                try:
                    start_time = time.time()
                    
                    # Parse context if provided
                    context = None
                    if mode == "With Context" and context_json:
                        try:
                            context = json.loads(context_json)
                        except:
                            context = offering_generator.get_default_context()
                    
                    # Generate offering
                    offering = offering_generator.generate_offering(
                        prompt=prompt,
                        use_context=(mode == "With Context"),
                        context=context,
                        use_schema=(mode == "Schema-Guided"),
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_p=top_p,
                        num_beams=num_beams
                    )
                    
                    gen_time = time.time() - start_time
                    
                    # Format output
                    offering_json = json.dumps(offering, indent=2, ensure_ascii=False)
                    
                    # Validation
                    validation_results = []
                    validation_results.append(" Valid JSON-LD structure")
                    
                    if "@context" in offering:
                        validation_results.append(f" @context present ({len(offering['@context'])} namespaces)")
                    else:
                        validation_results.append(" Missing @context")
                    
                    if "@graph" in offering:
                        entity_count = len(offering["@graph"])
                        validation_results.append(f" @graph present ({entity_count} entities)")
                    
                        # Check for required entity types
                        entity_types = set()
                        for entity in offering["@graph"]:
                            if "@type" in entity:
                                entity_types.add(entity["@type"])
                        
                        required_types = ["sedimark:Offering", "sedimark:Asset"]
                        for req_type in required_types:
                            if req_type in entity_types:
                                validation_results.append(f" {req_type} found")
                            else:
                                validation_results.append(f" {req_type} not found")
                    else:
                        validation_results.append(" Missing @graph")
                    
                    validation_text = "\n".join(validation_results)
                    
                    # Statistics
                    stats = f"""Generation Statistics:
Mode: {mode}
Generation Time: {gen_time:.2f}s
Output Size: {len(offering_json)} characters
Entities: {len(offering.get('@graph', []))}
Namespaces: {len(offering.get('@context', {}))}
Valid JSON:  Yes"""
                    
                    # Metrics
                    estimated_tokens = len(offering_json) // 4
                    tok_per_sec = f"{estimated_tokens / gen_time:.1f}" if gen_time > 0 else "N/A"
                    
                    tok_per_sec_html = f"<div class='metric-value'>{tok_per_sec}</div><div class='metric-label'>Tokens/Second</div>"
                    out_tokens_html = f"<div class='metric-value'>~{estimated_tokens}</div><div class='metric-label'>Output Tokens</div>"
                    gen_time_html = f"<div class='metric-value'>{gen_time:.2f}s</div><div class='metric-label'>Generation Time</div>"
                    json_valid_html = "<div class='metric-value'></div><div class='metric-label'>JSON Status</div>"
                    
                    return (offering_json, validation_text, stats,
                           tok_per_sec_html, out_tokens_html, gen_time_html, json_valid_html)
                    
                except Exception as e:
                    error_msg = f" Generation failed: {str(e)}"
                    return (error_msg, error_msg, error_msg,
                           "<div class='metric-value'>Error</div><div class='metric-label'>Tokens/Second</div>",
                           "<div class='metric-value'>Error</div><div class='metric-label'>Output Tokens</div>",
                           "<div class='metric-value'>Error</div><div class='metric-label'>Generation Time</div>",
                           "<div class='metric-value'>error</div><div class='metric-label'>JSON Status</div>")
            
            # Connect event handlers
            generation_mode.change(
                fn=update_mode_visibility,
                inputs=[generation_mode],
                outputs=[context_input]
            )
            
            clear_btn.click(
                fn=clear_prompt,
                outputs=[prompt_input]
            )
            
            example_btn.click(
                fn=load_example,
                outputs=[prompt_input]
            )
            
            prompt_input.change(
                fn=count_characters,
                inputs=[prompt_input],
                outputs=[char_count]
            )
            
            generate_button.click(
                fn=generate_offering_wrapper,
                inputs=[prompt_input, generation_mode, context_input, 
                       max_tokens_slider, temp_slider, top_p_slider, num_beams_slider],
                outputs=[offering_output, validation_output, stats_output,
                        tokens_per_sec, total_tokens, generation_time, json_valid]
            )
            
            save_button.click(
                fn=save_offering,
                inputs=[offering_output, save_filename],
                outputs=save_status
            )
            
        else:
            gr.HTML("""
            <div style="text-align: center; padding: 3rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; margin: 2rem 0;">
                <h2 style="color: #dc2626; margin-bottom: 1rem;"> Model Loading Failed</h2>
                <p style="color: #7f1d1d;">The model failed to load. Check the console output for detailed error messages.</p>
                <p style="color: #7f1d1d; font-size: 0.875rem; margin-top: 1rem;">
                    Common issues: Missing checkpoint files, CUDA memory issues, incompatible dependencies
                </p>
            </div>
            """)
    
    return interface

if __name__ == "__main__":
    print(f"\nStarting SEDIMARK Offering Generator Interface...")
    print(f"Config: {config_dir}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Config: quantization={CONFIG['quantization']['inference']['enabled']}, qlora={CONFIG['qlora']['enabled']}")
    
    # List checkpoint contents for debugging
    try:
        checkpoint_files = os.listdir(CHECKPOINT_PATH)
        print(f"Checkpoint contains: {checkpoint_files}")
    except Exception as e:
        print(f"Could not list checkpoint contents: {e}")
    
    # STEP 1: Load model in background
    print("\n" + "="*60)
    print("STEP 1: LOADING MODEL IN BACKGROUND")
    print("="*60)
    
    model_loaded_successfully = startup_model_loading()
    
    if not model_loaded_successfully:
        print("\nMODEL LOADING FAILED")
        print("Check the error messages above for details.")
        print("Common solutions:")
        print("1. Check vocabulary size mismatch")
        print("2. Verify checkpoint files are complete")
        print("3. Ensure sufficient GPU memory")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("MODEL LOADED SUCCESSFULLY")

    print("="*60)
    
    # STEP 2: Create and launch Gradio interface
    print("\nSTEP 2: CREATING GRADIO INTERFACE")
    
    interface = create_gradio_interface()
    

    if interface is None:
        print("Failed to create Gradio interface")
        sys.exit(1)
    
    # Get launch configuration
    print("\nLaunch Configuration:")
    port = input("Port (default 7860): ").strip()
    port = int(port) if port.isdigit() else 7864
    
    share = input("Create public link? (y/N): ").strip().lower() == 'y'
    
    print(f"\nLaunching interface on port {port}...")
    if share:
        print("Public link will be generated...")
    
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=not share,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\nInterface stopped by user")
    except Exception as e:
        print(f"\nLaunch failed: {e}")
        print("Check that the port is not already in use")
