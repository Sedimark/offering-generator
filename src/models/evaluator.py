import torch
import gc
import os
import json
import logging
import traceback
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig
)
from torch.nn import CrossEntropyLoss
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

from config.config import CONFIG
from src.utils.memory import MemoryManager, deep_cleanup
from src.utils.metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class JSONLLMEvaluator:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. A GPU is required.")
        
        self.device = torch.device("cuda")
        self.max_length = CONFIG["max_length"]
        self.memory_logger = self._setup_memory_logger()
        self.memory_manager = MemoryManager(target_usage=0.85, warning_threshold=0.92)
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(CONFIG["logs_dir"])
        
        self._setup_memory()
        self._initialize_model()
        
        # Track initial metrics after model initialization
        self._track_initial_metrics()
        
    def _setup_memory_logger(self):
        memory_logger = logging.getLogger('memory_tracker')
        memory_logger.setLevel(logging.INFO)
        
        os.makedirs(os.path.join(CONFIG["logs_dir"], "memory_logs"), exist_ok=True)
        fh = logging.FileHandler(os.path.join(CONFIG["logs_dir"], "memory_logs", 'memory_usage.log'))
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        memory_logger.addHandler(fh)
        
        return memory_logger

    def _setup_memory(self):
        deep_cleanup()
        
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3
            
            self.memory_logger.info(f"GPU {i} - Total Memory: {total_memory:.2f}GB")
            self.memory_logger.info(f"GPU {i} - Allocated Memory: {allocated_memory:.2f}GB")
            self.memory_logger.info(f"GPU {i} - Cached Memory: {cached_memory:.2f}GB")
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        torch.backends.cudnn.benchmark = True

    def _initialize_model(self):
        try:
            self.memory_logger.info("Initializing tokenizer...")
            
            # Initialize tokenizer properly for JSON generation
            self.tokenizer = AutoTokenizer.from_pretrained(
                CONFIG["model_name"],
                local_files_only=True,
                cache_dir=CONFIG["cache_dir"],
                model_max_length=self.max_length,
                padding_side="left",
                truncation_side="left",
                use_fast=True,
                trust_remote_code=True  # For Qwen models
            )
            
            # Set pad token properly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Comprehensive special tokens for JSON-LD
            special_tokens = {
                'additional_special_tokens': [
                    # Custom markers for model training
                    '<|ontology|>', '<|/ontology|>',
                    '<|schema|>', '<|/schema|>',
                    '<|context|>', '<|/context|>',
                    '<|json_output|>', '<|/json_output|>',
                    '[CONTEXT_EMBEDDED]', '[JSON_START]', '[JSON_END]',
                    
                    # JSON-LD specific tokens
                    '"@context"', '"@graph"', '"@id"', '"@type"',
                    '"@value"', '"@language"', '"@list"', '"@set"',
                    '"@reverse"', '"@index"', '"@base"', '"@vocab"',
                    '"@container"', '"@version"', '"@protected"',
                    
                    # RDF/OWL vocabulary prefixes and common terms
                    'rdf:type', 'rdfs:Class', 'rdfs:subClassOf', 
                    'rdfs:domain', 'rdfs:range', 'rdfs:label',
                    'rdfs:comment', 'rdfs:seeAlso', 'rdfs:isDefinedBy',
                    'rdfs:Literal', 'rdfs:Resource', 'rdfs:subPropertyOf',
                    
                    'owl:Class', 'owl:ObjectProperty', 'owl:DatatypeProperty',
                    'owl:NamedIndividual', 'owl:FunctionalProperty', 
                    'owl:InverseFunctionalProperty', 'owl:TransitiveProperty',
                    'owl:SymmetricProperty', 'owl:Ontology', 'owl:imports',
                    'owl:versionInfo', 'owl:equivalentClass', 'owl:disjointWith',
                    'owl:unionOf', 'owl:intersectionOf', 'owl:oneOf',
                    'owl:allValuesFrom', 'owl:someValuesFrom', 'owl:hasValue',
                    'owl:cardinality', 'owl:minCardinality', 'owl:maxCardinality',
                    'owl:inverseOf', 'owl:propertyChainAxiom',
                    
                    # XSD data types
                    'xsd:string', 'xsd:boolean', 'xsd:decimal', 'xsd:float',
                    'xsd:double', 'xsd:integer', 'xsd:long', 'xsd:int',
                    'xsd:short', 'xsd:byte', 'xsd:nonNegativeInteger',
                    'xsd:positiveInteger', 'xsd:nonPositiveInteger',
                    'xsd:negativeInteger', 'xsd:dateTime', 'xsd:date',
                    'xsd:time', 'xsd:duration', 'xsd:anyURI',
                    
                    # SEDIMARK specific vocabulary
                    'sedimark:Asset', 'sedimark:DataAsset', 'sedimark:AIModelAsset',
                    'sedimark:ServiceAsset', 'sedimark:OtherAsset', 'sedimark:Offering',
                    'sedimark:Self-Listing', 'sedimark:AssetProvision', 
                    'sedimark:AssetQuality', 'sedimark:Agreement',
                    'sedimark:OfferingContract', 'sedimark:Participant',
                    
                    # SEDIMARK properties
                    'sedimark:hasAsset', 'sedimark:hasOffering', 'sedimark:hasSelf-Listing',
                    'sedimark:hasAssetQuality', 'sedimark:hasOfferingContract',
                    'sedimark:belongsTo', 'sedimark:isProvidedBy', 'sedimark:references',
                    'sedimark:isListedBy', 'sedimark:algorithm', 'sedimark:category',
                    'sedimark:execution', 'sedimark:handlesStream', 'sedimark:inputFormat',
                    'sedimark:outputFormat', 'sedimark:serialization', 'sedimark:inputParameters',
                    
                    # DCAT vocabulary
                    'dcat:Dataset', 'dcat:Distribution', 'dcat:Catalog',
                    'dcat:Resource', 'dcat:DataService', 'dcat:theme',
                    'dcat:accessURL', 'dcat:downloadURL', 'dcat:mediaType',
                    'dcat:byteSize', 'dcat:temporalResolution', 'dcat:startDate',
                    'dcat:endDate', 'dcat:spatial', 'dcat:temporal',
                    'dcat:keyword', 'dcat:landingPage', 'dcat:qualifiedRelation',
                    'dcat:hadRole', 'dcat:servesDataset', 'dcat:endpointURL',
                    'dcat:endpointDescription', 'dcat:conformsTo',
                    
                    # DQV vocabulary
                    'dqv:QualityMeasurement', 'dqv:Metric', 'dqv:Dimension',
                    'dqv:Category', 'dqv:computedOn', 'dqv:isMeasurementOf',
                    'dqv:value', 'dqv:hasQualityMeasurement', 'dqv:inDimension',
                    'dqv:inCategory', 'dqv:expectedDataType',
                    
                    # DCTerms vocabulary
                    'dcterms:title', 'dcterms:description', 'dcterms:creator',
                    'dcterms:contributor', 'dcterms:publisher', 'dcterms:issued',
                    'dcterms:modified', 'dcterms:identifier', 'dcterms:language',
                    'dcterms:license', 'dcterms:rights', 'dcterms:format',
                    'dcterms:source', 'dcterms:subject', 'dcterms:temporal',
                    'dcterms:spatial', 'dcterms:created', 'dcterms:valid',
                    'dcterms:available', 'dcterms:conformsTo', 'dcterms:hasPart',
                    'dcterms:isPartOf', 'dcterms:hasVersion', 'dcterms:isVersionOf',
                    'dcterms:references', 'dcterms:isReferencedBy', 'dcterms:replaces',
                    'dcterms:isReplacedBy', 'dcterms:relation', 'dcterms:coverage',
                    'dcterms:audience', 'dcterms:accrualPeriodicity',
                    
                    # FOAF vocabulary
                    'foaf:Agent', 'foaf:Person', 'foaf:Organization', 'foaf:Document',
                    'foaf:OnlineAccount', 'foaf:name', 'foaf:homepage', 'foaf:mbox',
                    'foaf:openid', 'foaf:accountName', 'foaf:accountServiceHomepage',
                    'foaf:img', 'foaf:page', 'foaf:primaryTopic', 'foaf:logo',
                    
                    # SKOS vocabulary
                    'skos:Concept', 'skos:ConceptScheme', 'skos:prefLabel',
                    'skos:altLabel', 'skos:definition', 'skos:notation',
                    'skos:broader', 'skos:narrower', 'skos:related',
                    'skos:inScheme', 'skos:topConceptOf', 'skos:hasTopConcept',
                    'skos:exactMatch', 'skos:closeMatch', 'skos:relatedMatch',
                    'skos:broadMatch', 'skos:narrowMatch', 'skos:note',
                    'skos:scopeNote', 'skos:editorialNote', 'skos:changeNote',
                    
                    # ODRL vocabulary  
                    'odrl:Policy', 'odrl:Asset', 'odrl:Party', 'odrl:Action',
                    'odrl:Permission', 'odrl:Prohibition', 'odrl:Duty',
                    'odrl:Constraint', 'odrl:target', 'odrl:assignee',
                    'odrl:assigner', 'odrl:action', 'odrl:constraint',
                    'odrl:Agreement', 'odrl:Offer', 'odrl:Set',
                    
                    # PROV vocabulary
                    'prov:Activity', 'prov:Entity', 'prov:Agent',
                    'prov:wasGeneratedBy', 'prov:wasDerivedFrom',
                    'prov:wasAttributedTo', 'prov:used', 'prov:wasAssociatedWith',
                    'prov:actedOnBehalfOf', 'prov:wasInformedBy',
                    'prov:startedAtTime', 'prov:endedAtTime', 'prov:atLocation',
                    
                    # Common URI prefixes (not as tokens but useful patterns)
                    'http://www.w3.org/', 'https://w3id.org/', 'http://purl.org/',
                    'http://xmlns.com/', 'http://schema.org/',
                    
                    # Namespace declarations common in JSON-LD
                    '"@prefix"', '"@base"', '"@vocab"',
                    
                    # Blank node identifiers pattern
                    '_:genid', '_:b', '_:node',
                    
                    # Common JSON-LD keywords for framing
                    '"@explicit"', '"@default"', '"@embed"', '"@omitDefault"',
                    '"@requireAll"', '"@frameDefault"'
                ]
            }
            
            # Add special tokens from file if provided
            if CONFIG.get("special_tokens_file"):
                try:
                    with open(CONFIG["special_tokens_file"], 'r') as f:
                        additional_tokens = json.load(f)
                        if isinstance(additional_tokens, list):
                            special_tokens['additional_special_tokens'].extend(additional_tokens)
                        else:
                            # If the file contains the same structure as above
                            special_tokens = additional_tokens
                        logger.info(f"Loaded additional special tokens from file")
                except Exception as e:
                    logger.warning(f"Could not load special tokens from file: {e}")
            
            # Add tokens to tokenizer
            added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            self.memory_logger.info(f"Added {added_tokens} special tokens to tokenizer")
            
            # Log some token statistics
            total_special_tokens = len(special_tokens.get('additional_special_tokens', []))
            self.memory_logger.info(f"Total special tokens configured: {total_special_tokens}")
            
            # Verify some key tokens were added
            key_tokens_to_check = ['"@context"', 'owl:Class', 'dcat:Dataset', 'sedimark:Asset']
            for token in key_tokens_to_check:
                token_id = self.tokenizer.additional_special_tokens_ids[
                    special_tokens['additional_special_tokens'].index(token)
                ] if token in special_tokens['additional_special_tokens'] else None
                if token_id:
                    self.memory_logger.info(f"Token '{token}' successfully added with ID: {token_id}")
            
      
            
            # Add tokens to tokenizer
            added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            self.memory_logger.info(f"Added {added_tokens} special tokens to tokenizer")
            
            # Load model based on configuration
            use_quantization = CONFIG["quantization"]["inference"]["enabled"]
            use_qlora = CONFIG["qlora"]["enabled"]
            
            if use_quantization or use_qlora:
                self._load_quantized_model()
            else:
                self._load_standard_model()
            
            # Configure for memory efficiency
            if CONFIG["gradient_checkpointing"]:
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = False
                self.memory_logger.info("Enhanced gradient checkpointing enabled")
            
            deep_cleanup()
            
            # Resize embeddings to account for new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.memory_logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
            
            # Initialize loss function
            self.loss_fn = CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,
                label_smoothing=CONFIG["label_smoothing"]
            )
            
            self.memory_logger.info("Model initialization complete")
            self.memory_manager.enhance_memory_logging()
            
        except Exception as e:
            self.memory_logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _track_initial_metrics(self):
        """Track initial model metrics"""
        logger.info("Tracking initial model metrics...")
        self.metrics_tracker.update_metrics(
            self.model,
            self.tokenizer,
            is_baseline=True
        )
        logger.info("Initial metrics tracked successfully")

    def _load_quantized_model(self):
        """Load model with quantization"""
        self.memory_logger.info(f"Using quantization with {CONFIG['quantization']['inference']['bits']}-bit precision")
        
        # Setup quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = CONFIG["quantization"]["inference"]["bits"] == 4,
            load_in_8bit = CONFIG["quantization"]["inference"]["bits"] == 8,
            llm_int8_enable_fp32_cpu_offload = CONFIG.get("cpu_offloading", {}).get("enabled", False),                  
            bnb_4bit_compute_dtype = torch.bfloat16 
                                    if torch.cuda.is_bf16_supported() 
                                    else torch.float16,
            bnb_4bit_use_double_quant  = CONFIG["quantization"]["inference"]["double_quant"],
            bnb_4bit_quant_type        = CONFIG["quantization"]["inference"]["quant_type"],
        )

        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            local_files_only=True,
            cache_dir=CONFIG["cache_dir"],
            device_map="auto",
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            quantization_config=quantization_config
        )
        
        # Apply QLoRA if enabled
        if CONFIG["qlora"]["enabled"]:
            self._apply_qlora()

    def _apply_qlora(self):
        """Simplified QLoRA with guaranteed parameter enabling"""
        self.memory_logger.info("=== APPLYING QLORA (SIMPLIFIED) ===")
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Step 1: Prepare model
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Step 2: Simple LoRA config
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Step 3: Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Step 4: FORCE enable all LoRA parameters
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    trainable_count += param.numel()
                    self.memory_logger.info(f"✓ ENABLED: {name}")
            
            # Step 5: Verify we have trainable parameters
            if trainable_count == 0:
                raise RuntimeError("CRITICAL: No LoRA parameters found!")
            
            self.memory_logger.info(f" QLoRA SUCCESS: {trainable_count:,} trainable parameters")
            return True
            
        except Exception as e:
            self.memory_logger.error(f" QLoRA FAILED: {e}")
            return False


    def _load_standard_model(self):
        """Load model without quantization"""
        precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.memory_logger.info(f"Loading model with {precision_dtype} precision")
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            local_files_only=True,
            cache_dir=CONFIG["cache_dir"],
            torch_dtype=precision_dtype,
            device_map="auto",
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str = None):
        """Creates an evaluator instance from a checkpoint"""
        instance = cls.__new__(cls)  # Create instance without calling __init__
        
        # Initialize basic attributes
        instance.device = torch.device("cuda")
        instance.max_length = CONFIG["max_length"]
        instance.memory_logger = instance._setup_memory_logger()
        instance.memory_manager = MemoryManager(target_usage=0.85, warning_threshold=0.92)
        
        # Initialize metrics tracker
        instance.metrics_tracker = MetricsTracker(CONFIG["logs_dir"])
        
        try:
            checkpoint_path = checkpoint_dir or CONFIG["checkpoint_dir"]
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
                    
            instance.memory_logger.info(f"Loading from checkpoint: {checkpoint_path}")
            
            # Check for QLoRA adapter
            is_qlora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
            
            # Load tokenizer
            instance.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                cache_dir=CONFIG["cache_dir"],
                padding_side="left",
                trust_remote_code=True
            )
            
            # Set pad token
            if instance.tokenizer.pad_token is None:
                instance.tokenizer.pad_token = instance.tokenizer.eos_token
                instance.tokenizer.pad_token_id = instance.tokenizer.eos_token_id
            
            vocab_size = len(instance.tokenizer)
            instance.memory_logger.info(f"Checkpoint tokenizer vocabulary size: {vocab_size}")
            
            # Load model
            if is_qlora:
                instance._load_qlora_checkpoint(checkpoint_path, vocab_size)
            else:
                instance._load_standard_checkpoint(checkpoint_path, vocab_size)
            
            # Initialize loss function
            instance.loss_fn = CrossEntropyLoss(
                ignore_index=instance.tokenizer.pad_token_id,
                label_smoothing=CONFIG["label_smoothing"]
            )
            
            deep_cleanup()
            
            # Track metrics after loading checkpoint
            logger.info("Tracking checkpoint metrics...")
            instance.metrics_tracker.update_metrics(
                instance.model,
                instance.tokenizer
            )
            
            instance.memory_logger.info("Successfully loaded model from checkpoint")
            return instance
            
        except Exception as e:
            instance.memory_logger.error(f"Failed to load checkpoint: {str(e)}")
            instance.memory_logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _load_qlora_checkpoint(self, checkpoint_path: str, vocab_size: int):
        """Load QLoRA checkpoint with proper vocabulary handling"""
        self.memory_logger.info("Loading QLoRA checkpoint with vocabulary matching")
        
        try:
            # CRITICAL FIX: Load tokenizer first to get correct vocab size
            checkpoint_tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            actual_vocab_size = len(checkpoint_tokenizer)
            self.memory_logger.info(f"Checkpoint tokenizer vocab size: {actual_vocab_size}")
            
            # Load base model with quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model_name"],
                local_files_only=True,
                cache_dir=CONFIG["cache_dir"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config
            )
            
            # CRITICAL: Resize base model embeddings to match checkpoint
            current_vocab = base_model.get_input_embeddings().weight.shape[0]
            if current_vocab != actual_vocab_size:
                self.memory_logger.info(f"Resizing base model embeddings: {current_vocab} -> {actual_vocab_size}")
                base_model.resize_token_embeddings(actual_vocab_size)
            
            # Prepare for PEFT
            if CONFIG["quantization"]["inference"]["enabled"]:
                base_model = prepare_model_for_kbit_training(base_model)
            
            # Load PEFT adapter with size mismatch handling
            self.model = PeftModel.from_pretrained(
                base_model, 
                checkpoint_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                ignore_mismatched_sizes=True
            )
            
            # Update tokenizer to match checkpoint
            self.tokenizer = checkpoint_tokenizer
            
            self.memory_logger.info("Successfully loaded QLoRA checkpoint")
            
        except Exception as e:
            self.memory_logger.error(f"Failed to load QLoRA checkpoint: {e}")
            raise

    def _load_standard_checkpoint(self, checkpoint_path: str, vocab_size: int):
        """Load standard checkpoint"""
        self.memory_logger.info("Loading standard checkpoint")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                ignore_mismatched_sizes=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        except ValueError as e:
            if "Trying to set a tensor of shape" in str(e):
                self.memory_logger.warning(f"Embedding size mismatch detected: {str(e)}")
                self.memory_logger.info("Attempting fallback loading method...")
                
                # Load original model and resize
                self.model = AutoModelForCausalLM.from_pretrained(
                    CONFIG["model_name"],
                    local_files_only=True,
                    cache_dir=CONFIG["cache_dir"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto",
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True,
                )
                
                self.model.resize_token_embeddings(vocab_size)
            else:
                raise
        
        # Ensure embeddings match
        if self.model.get_input_embeddings().weight.shape[0] != vocab_size:
            self.model.resize_token_embeddings(vocab_size)
    
    # Add this method to your JSONLLMEvaluator class in evaluator.py

    @classmethod
    def from_checkpoint_with_peft(cls, checkpoint_dir: str = None):
        """Creates an evaluator instance from a PEFT/LoRA checkpoint"""
        instance = cls.__new__(cls)  # Create instance without calling __init__
        
        # Initialize basic attributes
        instance.device = torch.device("cuda")
        instance.max_length = CONFIG["max_length"]
        instance.memory_logger = instance._setup_memory_logger()
        instance.memory_manager = MemoryManager(target_usage=0.85, warning_threshold=0.92)
        
        # Initialize metrics tracker
        instance.metrics_tracker = MetricsTracker(CONFIG["logs_dir"])
        
        try:
            checkpoint_path = checkpoint_dir or CONFIG["checkpoint_dir"]
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
                    
            instance.memory_logger.info(f"Loading PEFT checkpoint from: {checkpoint_path}")
            
            # Check if this is a PEFT checkpoint
            is_peft = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
            
            if is_peft:
                instance.memory_logger.info("Detected PEFT/LoRA checkpoint")

                instance.tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint_path,
                    local_files_only=True,
                    cache_dir=CONFIG["cache_dir"],
                    padding_side="left",
                    trust_remote_code=True
                )

                # Set pad token
                if instance.tokenizer.pad_token is None:
                    instance.tokenizer.pad_token = instance.tokenizer.eos_token
                    instance.tokenizer.pad_token_id = instance.tokenizer.eos_token_id

                # Get ACTUAL vocab size from checkpoint
                vocab_size = len(instance.tokenizer)
                instance.memory_logger.info(f"Checkpoint tokenizer vocabulary size: {vocab_size}")
                
                # Load base model (the original model before fine-tuning)
                base_model_path = CONFIG["model_name"]  # Original base model path
                
                instance.memory_logger.info(f"Loading base model from: {base_model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    local_files_only=True,
                    cache_dir=CONFIG["cache_dir"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto",
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True,
                )
                
                # Resize base model embeddings to match tokenizer
                current_vocab = base_model.get_input_embeddings().weight.shape[0]
                if current_vocab != vocab_size:
                    instance.memory_logger.info(f"Resizing base model embeddings: {current_vocab} -> {vocab_size}")
                    base_model.resize_token_embeddings(vocab_size)
                
                # Load PEFT adapter
                from peft import PeftModel
                instance.memory_logger.info("Loading PEFT adapter...")
                instance.model = PeftModel.from_pretrained(
                    base_model, 
                    checkpoint_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                
                instance.memory_logger.info("Successfully loaded PEFT model")
                
            else:
                # Fallback to standard loading
                instance.memory_logger.info("No PEFT detected, using standard loading")
                return cls.from_checkpoint(checkpoint_path)
            
            # Initialize loss function
            instance.loss_fn = CrossEntropyLoss(
                ignore_index=instance.tokenizer.pad_token_id,
                label_smoothing=CONFIG["label_smoothing"]
            )
            
            # Track metrics after loading checkpoint
            logger.info("Tracking checkpoint metrics...")
            instance.metrics_tracker.update_metrics(
                instance.model,
                instance.tokenizer
            )
            
            instance.memory_logger.info("Successfully loaded PEFT model from checkpoint")
            return instance
            
        except Exception as e:
            instance.memory_logger.error(f"Failed to load PEFT checkpoint: {str(e)}")
            import traceback
            instance.memory_logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def prepare_for_json_generation(self, text: str) -> torch.Tensor:
        """Prepare text for JSON generation"""
        # Ensure proper formatting
        if not text.strip().endswith('<|json_output|>'):
            text += '\n<|json_output|>'
        
        # Tokenize with proper attention
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        return tokens