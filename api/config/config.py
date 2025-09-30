import os
import torch

# Base paths
BASE_DIR = "."
EXPERIMENT_DIR = os.path.join(BASE_DIR, "End_end_experimeny/Off_Gen_EXP/Exp_3B")

# Configuration dictionary with optimized parameters
CONFIG = {
    "model_name": os.path.join(BASE_DIR, "LLM_FineTuning/cache_directory/Qwen/Qwen2.5-3B"),
    #"model_name":  "Qwen/Qwen2.5-7B",
    "cache_dir": os.path.join(BASE_DIR, "LLM_FineTuning/cache_directory/Qwen/Qwen2.5-3B"),
    "teacher_output_dir": os.path.join(BASE_DIR, "Student_TeacherLearning/Teacher_Learning_GeneratedData/TrainingData_o1Mini_codehigh/"),
    "save_dir": EXPERIMENT_DIR,
    "context_file": "context.json",
    "output_dir": EXPERIMENT_DIR,
    "checkpoint_dir": os.path.join(EXPERIMENT_DIR, "checkpoints"),
    "best_model_dir": os.path.join(EXPERIMENT_DIR, "best_model"),
    "logs_dir": os.path.join(EXPERIMENT_DIR, "logs"),
    "ontology_file": "",
    "schema_file": "schema_structure.json",
   
    # Model parameters
    "max_length": 8192,  # Reduced from 8192 to improve memory efficiency
    "gradient_checkpointing": True,
    "batch_size": 1,
    "epochs": 20,
    "learning_rate": 2e-5,
    "min_learning_rate": 5e-7,
    "dataloader_num_workers": 2,  # Reduced from 8 to minimize memory pressure
    "grad_accum_steps": 64,
    "max_grad_norm": 0.5,
    "warmup_steps": 500,
    "warmup_ratio": 0.15,
    "weight_decay": 0.02,
    "save_steps": 100,
    "eval_steps": 50,
    "logging_steps": 20,
    "label_smoothing": 0.15,
    "early_stopping_patience": 5,  # Reduced from 20 to save training time
    "chunk_overlap": 64,  # New setting for chunk overlap
    "training_phase": "full_context",  # NEW: Default to full context training
    "context_dropout_rate": 0.0,
   
    # Generation parameters for JSON
    "generation_temperature": 0.1,  # Low temperature for structured output
    "generation_top_p": 0.95,
    "generation_repetition_penalty": 1.2,
    "generation_num_beams": 4,
    "generation_no_repeat_ngram_size": 3,
   
    # File patterns
    "teacher_output_file_pattern": "*.json",
   
    # Default test values
    
    "default_test_prompt": """Generate a structured JSON-LD document for the user input.

Use this exact @context:
{
  "sedi": "https://w3id.org/sedimark/ontology#",
  "dct": "http://purl.org/dc/terms/",
  "owl": "http://www.w3.org/2002/07/owl#",
  "dcat": "http://www.w3.org/ns/dcat#",
  "xsd": "http://www.w3.org/2001/XMLSchema#",
  "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
  "@vocab": "https://w3id.org/sedimark/ontology#"
}

Create @graph with exactly 8 entities:
1. [location-name] - Type: ["owl:NamedIndividual", "dct:Location"]
2. [data-type]-period - Type: ["owl:NamedIndividual", "dct:PeriodOfTime"]
3. [data-type]-measurements - Type: ["Self-Listing", "owl:NamedIndividual"]
4. [organization-name] - Type: ["owl:NamedIndividual", "Participant"]
5. [data-type]-offering-001 - Type: ["owl:NamedIndividual", "Offering"]
6. [data-type]-asset-001 - Type: ["owl:NamedIndividual", "DataAsset"]
7. [data-type]-asset-001-quality - Type: ["owl:NamedIndividual", "AssetQuality"]
8. [data-type]-contract - Type: ["owl:NamedIndividual", "OfferingContract"]

Base URI: https://sedimark.surrey.ac.uk/ecosystem/
Link entities using @id references.
Set temporal resolution based on update frequency.
Use ISO dates for startDate/endDate.
""",
    "default_test_input": """Air monitoring station Helsinki from 2025-01-01 to ongoing. Located at COORDINATES: 24.81117317, 60.2202584 recoeded daily.
ENDPOINT_URL: https://sedimark-helsinki.stellio.io/ngsi-ld/v1/temporal/entities/urn:sedimark:station:1
ENDPOINT_TYPE: NGSI-LD
ORGANIZATION: Helsinki Environmental Monitoring
ORGANIZATION_ID: https://sedimark.helsinki.fi/ecosystem/MyParticipantOrg
LICENSE: Creative Commons Attribution-NoDerivatives 4.0 International
LICENSE_URL: https://creativecommons.org/licenses/by-nd/4.0/
CREATED: 2025-01-01
MODIFIED: 2025-01-15
LANGUAGE: English
OFFERING_ID: https://sedimark.helsinki.fi/ecosystem/air-quality-offering-001
SELF_LISTING_ID: https://sedimark.helsinki.fi/ecosystem/helsinki-air-quality-lab
CATALOG_ID: https://sedimark.helsinki.fi/ecosystem/helsinki-local-catalogue""",
   
    # Context curriculum training
    "context_curriculum": {
        "enabled": True,
        "full_context_phase_ratio": 1.0,
        "partial_context_phase_ratio": 0.0,
        "context_dropout_rates": [0.0],
        "lr_decay_factor": 1.0,
        "chunk_schema": False
    },

    "full_context_training": {
    "sampling_weights": (0.7, 0.2, 0.1),  # Teacher, Ontology, Schema
    "task_weights": (0.7, 0.2, 0.1),      # Balanced for full context
    "context_dropout_rate": 0.0,           # Always use full context
    "use_context_embedding": True          # Always embed context
},
   
    # Mixed precision settings - Updated
    "mixed_precision": {
        "enabled": True,
        "dtype": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        "opt_level": "O2",
        "casting": True
    },
   
    # Checkpoint management
    "checkpoint_management": {
        "max_checkpoints_to_keep": 3,
        "checkpoint_cleanup_interval": 5
    },
   
    # Progressive data loading - Updated
    "progressive_loading": {
        "enabled": False,
        "cache_size": 32,  # Reduced from 100 to save memory
        "prefetch_factor": 2
    },
   
    # Quantization settings - Updated and Enabled
    "quantization": {
        "inference": {
            "enabled": True,  # Changed from False to True
            "bits": 4,
            "quant_type": "nf4",
            "double_quant": True,
        },
        "training": {
            "enabled": True,  # Changed from False to True
            "bits": 4,        # Changed from 8 to 4
            "quant_type": "nf4", # Changed from int8 to nf4
            "double_quant": True
        }
    },
   
    # QLoRA settings - Updated and Enabled
    "qlora": {
    "enabled": True,
    "r": 8,  # Reduced from 16
    "alpha": 16,  # Reduced from 32
    "dropout": 0.05,
    "target_modules": [
        "q_proj",  # Using only query projection 
        "v_proj"   # and value projection
    ]
},
    
    # CPU offloading - New
    "cpu_offloading": {
        "enabled": True,
        "offload_activations": True,
        "offload_parameters": True
    },
    
    # Debug settings - New
    "debug": {
        "log_memory_usage_frequency": 50,
        "log_gradients": True,
        "log_parameters": True,
        "verify_parameters": True,
        "exception_handling": {
            "retry_on_oom": True,
            "skip_problematic_batches": True,
            "log_traceback": True
        }
    }
}
