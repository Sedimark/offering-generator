# Testing Commands for SEDIMARK Offering Generator

## Basic Testing

### Test with Default Configuration

python SEDIMARK_OFFERING/scripts/test.py


### Test with Best Model

python SEDIMARK_OFFERING/scripts/test.py --use_best_model --validate


### Test Specific Checkpoint

python SEDIMARK_OFFERING/scripts/test.py --checkpoint checkpoints/checkpoint-500 --validate


## Context Testing

### Test with Context

python SEDIMARK_OFFERING/scripts/test.py --use_context --validate


### Test with Schema-Guided Generation

python SEDIMARK_OFFERING/scripts/test.py --use_schema --validate


## Custom Input Testing

### Test with Custom Input

python SEDIMARK_OFFERING/scripts/test.py --test_input "Generate JSON-LD for temperature sensor data from 50 IoT devices in London" --validate


### Test with Input File

python SEDIMARK_OFFERING/scripts/test.py --test_file data/test_samples/custom_input.txt --validate


## Model Comparison Testing

### Test Different Checkpoints

# Test standard model
python SEDIMARK_OFFERING/scripts/test.py --checkpoint experiments/training_modes/standard --output_dir outputs/test/standard --validate

# Test full context model  
python SEDIMARK_OFFERING/scripts/test.py --checkpoint experiments/training_modes/full_context --use_context --output_dir outputs/test/full_context --validate

# Test QLoRA model
python SEDIMARK_OFFERING/scripts/test.py --checkpoint experiments/quantization/qlora_4bit --output_dir outputs/test/qlora --validate


## Production Testing

### Test Production Model

python SEDIMARK_OFFERING/scripts/test.py --checkpoint production/final_model --use_schema --validate --output_dir outputs/production


### Test with Optimal Generation Parameters

python SEDIMARK_OFFERING/scripts/test.py --use_best_model --temperature 0.1 --top_p 0.95 --num_beams 4 --validate
