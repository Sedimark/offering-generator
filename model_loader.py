import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import CONFIG

logger = logging.getLogger(__name__)

class ModelLoader:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self, checkpoint_path: str = None):
        checkpoint_path = checkpoint_path or CONFIG["checkpoint_path"]

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from: {checkpoint_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            padding_side="left",
            trust_remote_code=True
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Resize embeddings to match checkpoint tokenizer
        vocab_size = len(self.tokenizer)
        if base_model.get_input_embeddings().weight.shape[0] != vocab_size:
            logger.info(f"Resizing embeddings to {vocab_size}")
            base_model.resize_token_embeddings(vocab_size)

        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        logger.info("Model loaded successfully")
        return self.model, self.tokenizer

    def prepare_inputs(self, text: str):
        if not text.strip().endswith('<|json_output|>'):
            text += '\n<|json_output|>'

        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_attention_mask=True
        ).to(self.device)

    def generate(self, inputs, **generation_params):
        # Merge with default generation params
        params = {**CONFIG["generation"], **generation_params}
        params["pad_token_id"] = self.tokenizer.pad_token_id
        params["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **params
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
