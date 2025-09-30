# Training Commands for Experiments

## 1. QUANTIZATION COMPARISON

### Full Precision (Baseline - High Memory)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization none --output_dir experiments/quantization/full_precision

### 8-bit Quantization (INT8)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 8bit --output_dir experiments/quantization/int8

### 4-bit Quantization (QLoRA)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --output_dir experiments/quantization/qlora_4bit

## 2. LORA RANK COMPARISON

### Small LoRA (r=4)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --lora_r 4 --lora_alpha 8 --output_dir experiments/lora_sizes/r4

### Standard LoRA (r=8)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --lora_r 8 --lora_alpha 16 --output_dir experiments/lora_sizes/r8

### Large LoRA (r=16)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --lora_r 16 --lora_alpha 32 --output_dir experiments/lora_sizes/r16

## 3. TRAINING MODE COMPARISON

### Standard Training
python SEDIMARK_OFFERING/scripts/train.py --mode standard --epochs 20 --quantization 4bit --use_qlora --output_dir experiments/training_modes/standard

### Single Phase - Full Context Only
python SEDIMARK_OFFERING/scripts/train.py --mode single_phase --phase full_context --epochs 15 --quantization 4bit --use_qlora --output_dir experiments/training_modes/full_context

### Single Phase - Context-less
python SEDIMARK_OFFERING/scripts/train.py --mode single_phase --phase context_less --epochs 15 --quantization 4bit --use_qlora --output_dir experiments/training_modes/context_less

### Curriculum Learning (if time permits)
python SEDIMARK_OFFERING/scripts/train.py --mode curriculum --epochs 45 --quantization 4bit --use_qlora --output_dir experiments/training_modes/curriculum

## 4. OPTIMIZATION TECHNIQUES COMPARISON

### Baseline (No optimizations)
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --output_dir experiments/baseline/no_opt

### With Gradient Checkpointing
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --gradient_checkpointing --output_dir experiments/baseline/grad_checkpoint

### With All Optimizations
python SEDIMARK_OFFERING/scripts/train.py --mode quick --epochs 5 --quantization 4bit --use_qlora --gradient_checkpointing --cpu_offload --output_dir experiments/baseline/all_opt

## 5. PRODUCTION READY COMMANDS

### Best configuration (based on experiments)
python SEDIMARK_OFFERING/scripts/train.py --mode standard --epochs 20 --use_qlora --gradient_checkpointing --cpu_offload --lora_r 8 --lora_alpha 16 --quantization 4bit --learning_rate 2e-5 --grad_accum_steps 64 --output_dir production/final_model

### Extended training for best quality
python SEDIMARK_OFFERING/scripts/train.py --mode curriculum --epochs 45 --use_qlora --gradient_checkpointing --cpu_offload --lora_r 8 --lora_alpha 16 --quantization 4bit --learning_rate 1e-5 --warmup_steps 1000 --output_dir production/best_quality

## Production Ready Command (Reference)
python SEDIMARK_OFFERING/scripts/train.py --mode standard --epochs 20 --use_qlora --gradient_checkpointing --cpu_offload --lora_r 8 --lora_alpha 16 --quantization 4bit