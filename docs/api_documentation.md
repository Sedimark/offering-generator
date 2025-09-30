Installation

Ensure all dependencies are installed:

bashpip install fastapi uvicorn torch transformers peft pydantic jwt

Verify checkpoint files exist:

ls ./checkpoints/
# Should contain: adapter_config.json, adapter_model.safetensors, tokenizer.json

Starting the API Server

# Basic Start
python ./api_sedimark.py

# Start with Custom Port
python ./api_sedimark.py --port 8080

# Start with Environment Variables
export API_PORT=8080
export JWT_SECRET="your-secure-secret-key"
python ./api_sedimark.py

# Run in Background with Logging
nohup python api.py > api.log 2>&1 &
Run with Screen (Recommended for Long Sessions)
screen -S sedimark-api
python ./api_sedimark.py


# If run on cluster 
in new window
ssh aisurrey_submit01
ssh NODELIST

Check health: curl -X GET "http://localhost:8080/health"
** Use Post request:**

curl -X POST "http://localhost:8082/test"      -H "Content-Type: application/json"      -d @/mnt/fast/nobackup/scratch4weeks/ma04503/Sedimark/SEDIMARK_OFFERING/api/request.json      -s | jq -r '.text' > /mnt/fast/nobackup/scratch4weeks/ma04503/Sedimark/Api_outputs/greenhouse_output.json

# For Gardio api:
python ./api/gradio_api_single_prompt.py
