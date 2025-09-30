#!/usr/bin/env python3
"""
SEDIMARK-Compatible Offering Generator API
Production implementation for SEDIMARK offering management system
"""

# Import statements
import os
import sys
import logging
import json
import gc
import torch
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

#Set up paths BEFORE any project imports
config_dir = "."
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)  # Use insert(0) to prioritize this path

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import project modules 
try:
    from src.models.evaluator import JSONLLMEvaluator
    from config.config import CONFIG
    print("Successfully imported JSONLLMEvaluator and CONFIG")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    # Try alternative import
    sys.path.insert(0, os.path.join(config_dir, "src"))
    from models.evaluator import JSONLLMEvaluator
    from config.config import CONFIG


try:
    from src.utils.json_utils import validate_json_output, load_json_file
except ImportError:
    logging.warning("json_utils not available, using basic validation")
    
    def validate_json_output(output_str):
        """Basic JSON validation fallback"""
        try:
            json.loads(output_str)
            return output_str, True
        except json.JSONDecodeError:
            # Try to fix common issues
            if output_str.strip().endswith(','):
                output_str = output_str.strip()[:-1]
            try:
                json.loads(output_str)
                return output_str, True
            except:
                return output_str, False
    
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

try:
    from src.utils.memory import deep_cleanup
except ImportError:
    logging.warning("memory utils not available, using basic cleanup")
    
    def deep_cleanup():
        """Basic memory cleanup fallback"""
        torch.cuda.empty_cache()
        gc.collect()

print(f"Loaded config from: {config_dir}/config/config.py")
print(f"Base model path: {CONFIG['model_name']}")
print(f"Cache directory: {CONFIG['cache_dir']}")

# Set output directory for saved offerings
OUTPUT_DIR = Path("/mnt/f//SEDIMARK/Api_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure for PEFT/LoRA loading
CONFIG["quantization"]["inference"]["enabled"] = False
CONFIG["quantization"]["inference"]["load_in_4bit"] = False
CONFIG["quantization"]["inference"]["bnb_4bit_use_double_quant"] = False
CONFIG["qlora"]["enabled"] = True  # Enable for loading LoRA adapters

# Checkpoint path
CHECKPOINT_PATH = "Model_weights/Exp_3B/checkpoints/checkpoint-40"

print(f"Using checkpoint path: {CHECKPOINT_PATH}")

# Verify the checkpoint files exist
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

# Verify the checkpoint
verify_peft_checkpoint(CHECKPOINT_PATH)

print(f"Checkpoint contains tokenizer: {CHECKPOINT_PATH}")
print(f"Base model will be loaded from: {CONFIG['model_name']}")

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", "SEDIMARK")
if not JWT_SECRET or JWT_SECRET == "your-secret-key-here":
    print("Warning: Using default JWT secret. Set JWT_SECRET environment variable for production.")

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app - SEDIMARK API
app = FastAPI(
    title="SEDIMARK Offering Generator API",
    description="REST API for generating SEDIMARK-compliant offerings in JSON-LD format",
    version="1.0.0"
)

# Security dependency
auth_scheme = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    token = credentials.credentials
    import jwt
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return payload

# Request and response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    user_input: Optional[str] = Field(None, description="User input for offering generation")
    max_new_tokens: Optional[int] = Field(8192, ge=1, le=8192)
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(55, ge=1, le=100)
    repetition_penalty: Optional[float] = Field(1.2, ge=1.0, le=3.0)
    num_beams: Optional[int] = Field(4, ge=1, le=10)
    early_stopping: Optional[bool] = Field(True)
    length_penalty: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    no_repeat_ngram_size: Optional[int] = Field(3, ge=0, le=10)
    do_sample: Optional[bool] = Field(True)
    test_without_context: Optional[bool] = Field(False)
    use_context: Optional[bool] = Field(False)
    context: Optional[Dict[str, Any]] = Field(None, description="Context dictionary")
    use_schema: Optional[bool] = Field(False, description="Use schema-guided generation")
    pad_token_id: Optional[int] = Field(None)
    eos_token_id: Optional[int] = Field(None)

class GenerateResponse(BaseModel):
    text: str
    model_info: dict
    generation_time: Optional[float] = None
    tokens_generated: Optional[int] = None
    json_valid: Optional[bool] = None
    validation_results: Optional[Dict[str, Any]] = None

class ContextGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    context: Optional[Dict[str, Any]] = Field(None, description="Context dictionary")
    use_schema: Optional[bool] = Field(False, description="Use schema reference")
    max_new_tokens: Optional[int] = Field(8192)
    temperature: Optional[float] = Field(0.1)

# SEDIMARK Offering Generator class
class SEDIMARKOfferingGenerator:
    """Handles SEDIMARK-specific offering generation matching testing script behavior"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.schema_data = None
        self.load_schema()
        
    def load_schema(self):
        """Load schema file if available"""
        try:
            schema_file = CONFIG.get("schema_file")
            if schema_file and os.path.exists(schema_file):
                self.schema_data = load_json_file(schema_file)
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
    
    def generate_offering(self, 
                         prompt: str = None, 
                         user_input: str = None,
                         use_context: bool = False,
                         context: Dict[str, Any] = None,
                         use_schema: bool = False) -> Dict[str, Any]:
        """Generate a SEDIMARK-compliant offering matching testing script behavior"""
        
        input_text = user_input or prompt
        if not input_text:
            input_text = "Generate a SEDIMARK offering for IoT sensor data"
        
        # Format prompt based on context and schema options (matching testing script)
        if use_context and context:
            # Include context in input like test_with_checkpoint
            formatted_prompt = f"""{input_text}



{json.dumps(context, ensure_ascii=False)}



Generate a SEDIMARK-compliant JSON-LD offering with the following structure:
1. @graph array containing all entities
2. @context object at the end
3. Include: Offering, Asset, AssetQuality, AssetProvision, OfferingContract, SelfListing
4. Use proper @id references between entities
5. Follow SEDIMARK ontology strictly

<|json_output|>"""
        elif use_schema and self.schema_data:
            # Use schema-guided generation like test_with_schema_reference
            formatted_prompt = f"""{input_text}

SCHEMA REFERENCE:
{json.dumps(self.schema_data, indent=2, ensure_ascii=False)}

Instructions: Generate JSON-LD following the exact schema structure provided above.
1. Use the exact class definitions from the schema @graph
2. Follow the @context namespaces exactly as defined
3. Ensure all entities derive from owl:NamedIndividual as specified in the schema
4. Use the exact property names defined in rdfs:properties for each class

<|json_output|>"""
        else:
            # Use context-less format with [CONTEXT_EMBEDDED] token
            formatted_prompt = f"""{input_text}
[CONTEXT_EMBEDDED]


Generate a SEDIMARK-compliant JSON-LD offering with the following structure:
1. @graph array containing all entities
2. @context object at the end
3. Include: Offering, Asset, AssetQuality, AssetProvision, OfferingContract, SelfListing
4. Use proper @id references between entities
5. Follow SEDIMARK ontology strictly

<|json_output|>"""
        
        # Prepare input for generation (matching testing script)
        if hasattr(self.evaluator, 'prepare_for_json_generation'):
            inputs = self.evaluator.prepare_for_json_generation(formatted_prompt).to(self.evaluator.device)
        else:
            inputs = self.evaluator.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.evaluator.max_length
            ).to(self.evaluator.device)
        
        # Generation parameters matching testing script exactly
        generation_params = {
            'max_new_tokens': 8192,
            'do_sample': True,
            'temperature': 0.10,  # Low temperature for structured output
            'top_p': 0.90,
            'top_k': 55,
            'repetition_penalty': 1.2,
            'num_beams': 4,
            'early_stopping': True,
            'pad_token_id': self.evaluator.tokenizer.pad_token_id,
            'eos_token_id': self.evaluator.tokenizer.eos_token_id,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
        }
        
        # Use schema-specific parameters if using schema
        if use_schema:
            generation_params.update({
                'temperature': 0.1,
                'top_p': 0.95,
                'repetition_penalty': 2.0,
                'num_beams': 6,
                'length_penalty': 1.2,
            })
        
        with torch.inference_mode():
            try:
                # Clear cache before generation (matching testing)
                deep_cleanup()
                
                # Use mixed precision for generation (matching testing)
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
                    
                    # Reduce parameters for retry (matching testing fallback)
                    generation_params.update({
                        'num_beams': 1,  # Use greedy decoding as fallback
                        'max_new_tokens': 4096,
                        'do_sample': False,  # Disable sampling
                    })
                    
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = self.evaluator.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            **generation_params
                        )
                else:
                    raise
        
        # Decode output (matching testing)
        raw_output = self.evaluator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from generated text (matching testing)
        if "<|json_output|>" in raw_output:
            json_text = raw_output.split("<|json_output|>")[-1].strip()
        else:
            json_text = raw_output.strip()
        
        # Validate and fix JSON output (matching testing)
        json_text, is_valid = validate_json_output(json_text)
        
        if not is_valid:
            logger.warning("Generated output is not valid JSON, using fallback...")
            return self.generate_fallback_offering()
        
        # Parse JSON
        try:
            offering = json.loads(json_text)
            # Ensure proper structure
            if "@context" not in offering:
                offering["@context"] = self.get_default_context()
            
            # Validate if using schema
            if use_schema and self.schema_data:
                validation_results = self.validate_against_schema(offering)
                offering["_validation"] = validation_results
            
            return offering
        except json.JSONDecodeError:
            return self.generate_fallback_offering()
    
    def validate_against_schema(self, offering: Dict[str, Any]) -> Dict[str, Any]:
        """Validate offering against schema (matching testing script)"""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_checks": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check @context
            if "@context" in offering and "@context" in self.schema_data:
                schema_contexts = set(self.schema_data["@context"].keys())
                output_contexts = set(offering["@context"].keys()) if isinstance(offering["@context"], dict) else set()
                
                missing_contexts = schema_contexts - output_contexts
                if missing_contexts:
                    validation_results["warnings"].append(f"Missing context prefixes: {list(missing_contexts)}")
                
                validation_results["validation_checks"]["context_coverage"] = len(output_contexts) / len(schema_contexts) if schema_contexts else 0
            
            # Check @graph entities
            if "@graph" in offering and "@graph" in self.schema_data:
                schema_classes = {item.get("@id") for item in self.schema_data["@graph"] if "@id" in item}
                output_entities = []
                
                for entity in offering["@graph"]:
                    if "@type" in entity:
                        entity_types = entity["@type"] if isinstance(entity["@type"], list) else [entity["@type"]]
                        output_entities.extend(entity_types)
                
                used_schema_classes = set(output_entities) & schema_classes
                validation_results["validation_checks"]["schema_class_usage"] = len(used_schema_classes) / len(schema_classes) if schema_classes else 0
                
                if len(used_schema_classes) < len(schema_classes) * 0.5:
                    validation_results["warnings"].append("Less than 50% of schema classes were used in output")
            
            # Check for required properties
            validation_results["validation_checks"]["has_context"] = "@context" in offering
            validation_results["validation_checks"]["has_graph"] = "@graph" in offering
            validation_results["validation_checks"]["is_valid_json"] = True
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
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

# Global variables
evaluator = None
offering_generator = None

@app.on_event("startup")
async def startup_event():
    global evaluator, offering_generator
    try:
        logger.info("Starting API server...")
        logger.info(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
        
        # Load model with existing logic
        logger.info(f"Loading model from PEFT checkpoint: {CHECKPOINT_PATH}")
        
        required_files = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(CHECKPOINT_PATH, f))]
        
        if missing_files:
            logger.error(f"Missing PEFT files: {missing_files}")
            raise RuntimeError(f"Incomplete PEFT checkpoint - missing: {missing_files}")
        
        logger.info(f"PEFT checkpoint verified")
        
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
        
        # Load evaluator
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
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model info:")
        logger.info(f"   - Vocabulary size: {vocab_size:,}")
        logger.info(f"   - Parameters: {param_count:,}")
        logger.info(f"   - Device: {evaluator.device}")
        logger.info(f"   - Max length: {evaluator.max_length}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        
        # Test tokenizer
        test_text = "This is a test"
        tokens = evaluator.tokenizer(test_text, return_tensors="pt")
        logger.info(f"Tokenizer test: '{test_text}' -> {tokens['input_ids'].shape[1]} tokens")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# SEDIMARK-compatible endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SEDIMARK Offering Generator",
        "version": "1.0.0",
        "endpoints": {
            "generate_offerings": "/api/offerings/generate",
            "generate_with_context": "/api/offerings/generate-with-context",
            "generate_with_schema": "/api/offerings/generate-with-schema",
            "test": "/test",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/api/offerings/generate", 
         response_class=JSONResponse,
         summary="Generate Sample Offerings",
         description="Generates sample SEDIMARK-compliant offerings in JSON-LD format")
async def generate_offerings(
    count: int = Query(1, ge=1, le=10, description="Number of offerings to generate"),
    prompt: Optional[str] = Query(None, description="Custom prompt for offering generation"),
    use_context: bool = Query(False, description="Use context in generation"),
    use_schema: bool = Query(False, description="Use schema-guided generation"),
    save: bool = Query(True, description="Save output to file")
):
    """
    Generate SEDIMARK-compliant offerings in JSON-LD format.
    Compatible with: https://github.com/Sedimark/offering-manager
    """
    
    if offering_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if count == 1:
            # Generate single offering
            offering = offering_generator.generate_offering(
                prompt=prompt,
                use_context=use_context,
                use_schema=use_schema
            )
            generation_time = time.time() - start_time
            
            # Add metadata
            offering["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "generation_time_seconds": round(generation_time, 2),
                "model": "SEDIMARK-Qwen2.5-3B",
                "use_context": use_context,
                "use_schema": use_schema
            }
            
            # Save to file if requested
            if save:
                output_file = OUTPUT_DIR / f"offering_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(offering, f, indent=2)
                logger.info(f"Offering saved to: {output_file}")
                offering["_metadata"]["saved_to"] = str(output_file)
            
            return JSONResponse(content=offering, media_type="application/ld+json")
        else:
            # Generate multiple offerings
            all_entities = []
            
            for i in range(count):
                offering = offering_generator.generate_offering(
                    prompt=prompt,
                    use_context=use_context,
                    use_schema=use_schema
                )
                if "@graph" in offering:
                    all_entities.extend(offering["@graph"])
            
            combined_offering = {
                "@graph": all_entities,
                "@context": offering_generator.get_default_context(),
                "_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generation_time_seconds": round(time.time() - start_time, 2),
                    "count": count,
                    "model": "SEDIMARK-Qwen2.5-3B",
                    "use_context": use_context,
                    "use_schema": use_schema
                }
            }
            
            # Save to file if requested
            if save:
                output_file = OUTPUT_DIR / f"offerings_{count}x_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(combined_offering, f, indent=2)
                logger.info(f"Offerings saved to: {output_file}")
                combined_offering["_metadata"]["saved_to"] = str(output_file)
            
            return JSONResponse(content=combined_offering, media_type="application/ld+json")
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/offerings/generate-with-context",
          response_class=JSONResponse,
          summary="Generate with Context",
          description="Generate offering with context like testing script")
async def generate_with_context(request: ContextGenerateRequest):
    """Generate offering with context matching testing script behavior"""
    
    if offering_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Generate with context
        offering = offering_generator.generate_offering(
            prompt=request.prompt,
            use_context=request.context is not None,
            context=request.context,
            use_schema=request.use_schema
        )
        
        generation_time = time.time() - start_time
        
        # Add metadata
        offering["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "generation_time_seconds": round(generation_time, 2),
            "had_context": request.context is not None,
            "use_schema": request.use_schema
        }
        
        return JSONResponse(content=offering, media_type="application/ld+json")
        
    except Exception as e:
        logger.error(f"Context generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/offerings/generate-with-schema",
          response_class=JSONResponse,
          summary="Generate with Schema",
          description="Generate offering with schema reference like testing script")
async def generate_with_schema(
    prompt: str = "Generate a SEDIMARK offering",
    user_input: Optional[str] = None
):
    """Generate offering with schema reference matching testing script behavior"""
    
    if offering_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Generate with schema
        offering = offering_generator.generate_offering(
            prompt=prompt,
            user_input=user_input,
            use_schema=True
        )
        
        generation_time = time.time() - start_time
        
        # Add metadata
        offering["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "generation_time_seconds": round(generation_time, 2),
            "use_schema": True,
            "schema_loaded": offering_generator.schema_data is not None
        }
        
        return JSONResponse(content=offering, media_type="application/ld+json")
        
    except Exception as e:
        logger.error(f"Schema generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/api/offerings/list-generated")
async def list_generated_offerings():
    """List all generated offerings in the output directory"""
    try:
        files = sorted(OUTPUT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        offerings_list = []
        
        for file in files[:20]:  # Limit to last 20 files
            stat = file.stat()
            offerings_list.append({
                "filename": file.name,
                "path": str(file),
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "url": f"/api/offerings/file/{file.name}"
            })
        
        return {
            "output_directory": str(OUTPUT_DIR),
            "total_files": len(files),
            "recent_offerings": offerings_list
        }
    except Exception as e:
        logger.error(f"Failed to list offerings: {e}")
        return {"error": str(e)}

@app.get("/api/offerings/file/{filename}")
async def get_generated_file(filename: str):
    """Retrieve a specific generated offering file"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
        
        return JSONResponse(content=content, media_type="application/ld+json")
    except Exception as e:
        logger.error(f"Failed to read file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

# Original endpoints (updated to match testing behavior)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generation endpoint matching testing script behavior"""
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Generating response for prompt length: {len(request.prompt)}")
        
        # Format input matching testing script
        if request.use_context and request.context:
            # Include context in input like test_with_checkpoint
            formatted_prompt = f"""{request.prompt}



{json.dumps(request.context, ensure_ascii=False)}



{request.user_input or ''}


<|json_output|>"""
        elif request.test_without_context or not request.use_context:
            # Use context-less format with [CONTEXT_EMBEDDED] token
            formatted_prompt = f"""{request.prompt}
[CONTEXT_EMBEDDED]


{request.user_input or ''}


<|json_output|>"""
        else:
            # Standard format
            formatted_prompt = f"{request.prompt}\n\n{request.user_input or ''}\n\n<|json_output|>"
        
        # Prepare input for generation (matching testing)
        if hasattr(evaluator, 'prepare_for_json_generation'):
            inputs = evaluator.prepare_for_json_generation(formatted_prompt).to(evaluator.device)
        else:
            inputs = evaluator.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=evaluator.max_length
            ).to(evaluator.device)
        
        logger.info(f"Input tokens: {inputs.input_ids.shape[1]}")
        
        # Generation parameters matching testing script
        generation_params = {
            'max_new_tokens': request.max_new_tokens,
            'do_sample': request.do_sample,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'repetition_penalty': request.repetition_penalty,
            'num_beams': request.num_beams,
            'early_stopping': request.early_stopping,
            'pad_token_id': request.pad_token_id or evaluator.tokenizer.pad_token_id,
            'eos_token_id': request.eos_token_id or evaluator.tokenizer.eos_token_id,
            'length_penalty': request.length_penalty,
            'no_repeat_ngram_size': request.no_repeat_ngram_size,
        }
        
        # Generate with memory management and mixed precision (matching testing)
        with torch.inference_mode():
            try:
                # Clear cache before generation
                deep_cleanup()
                
                # Use mixed precision
                precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast('cuda', dtype=precision_dtype):
                    outputs = evaluator.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **generation_params
                    )
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM error, reducing generation parameters...")
                    deep_cleanup()
                    
                    # Fallback parameters
                    generation_params.update({
                        'num_beams': 1,
                        'max_new_tokens': min(request.max_new_tokens, 4096),
                        'do_sample': False,
                    })
                    
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = evaluator.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            **generation_params
                        )
                else:
                    raise
        
        # Decode output (matching testing)
        raw_output = evaluator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (matching testing)
        if "<|json_output|>" in raw_output:
            generated_output = raw_output.split("<|json_output|>")[-1].strip()
        else:
            generated_output = raw_output.strip()
        
        # Validate JSON output
        generated_output, is_valid = validate_json_output(generated_output)
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        
        # Model info
        model_info = {
            "checkpoint_path": CHECKPOINT_PATH,
            "vocab_size": len(evaluator.tokenizer),
            "max_length": evaluator.max_length,
            "input_tokens": inputs.input_ids.shape[1],
            "output_tokens": tokens_generated,
            "generation_params": generation_params,
            "json_valid": is_valid,
            "use_context": request.use_context,
            "has_context": request.context is not None
        }
        
        # Basic validation results
        validation_results = None
        if is_valid:
            try:
                output_json = json.loads(generated_output)
                validation_results = {
                    "has_context": "@context" in output_json,
                    "has_graph": "@graph" in output_json,
                    "entity_count": len(output_json.get("@graph", [])) if "@graph" in output_json else 0
                }
            except:
                pass
        
        logger.info(f"Generation completed - output length: {len(generated_output)}, valid JSON: {is_valid}")
        
        return GenerateResponse(
            text=generated_output, 
            model_info=model_info,
            generation_time=generation_time,
            tokens_generated=tokens_generated,
            json_valid=is_valid,
            validation_results=validation_results
        )
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        deep_cleanup()
        raise HTTPException(status_code=500, detail="GPU memory error - try reducing max_new_tokens")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Test endpoint (no auth required)
@app.post("/test")
async def test_generate(request: GenerateRequest):
    """Test endpoint without authentication for development"""
    return await generate(request)

# Health check endpoint
@app.get("/health")
def health():
    global evaluator, offering_generator
    
    status_info = {
        "status": "healthy" if evaluator is not None else "error",
        "model_loaded": evaluator is not None,
        "offering_generator_loaded": offering_generator is not None,
        "checkpoint_path": CHECKPOINT_PATH,
        "output_directory": str(OUTPUT_DIR)
    }
    
    if evaluator is not None:
        status_info.update({
            "vocab_size": len(evaluator.tokenizer),
            "max_length": evaluator.max_length,
            "device": str(evaluator.device),
            "model_name": os.path.basename(CHECKPOINT_PATH)
        })
    
    if offering_generator is not None:
        status_info["schema_loaded"] = offering_generator.schema_data is not None
        if offering_generator.schema_data:
            status_info["schema_classes"] = len(offering_generator.schema_data.get("@graph", []))
    
    return status_info

if __name__ == "__main__":
    print("Starting SEDIMARK Offering Generator API...")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Config: quantization={CONFIG['quantization']['inference']['enabled']}, qlora={CONFIG['qlora']['enabled']}")
    
    # List checkpoint contents for debugging
    try:
        checkpoint_files = os.listdir(CHECKPOINT_PATH)
        print(f"Checkpoint contains: {checkpoint_files}")
    except Exception as e:
        print(f"Could not list checkpoint contents: {e}")
    
    # Fix: Use the actual filename without .py extension
    uvicorn.run(
        "api_sedimark:app",  # Use the actual module name
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8082)),
        workers=1,
        loop="uvloop",
        timeout_keep_alive=60,
        reload=False
    )
