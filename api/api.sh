#!/bin/bash

# Configuration
API_HOST="localhost"
API_PORT="8081"  # Your current port
BASE_URL="http://${API_HOST}:${API_PORT}"
OUTPUT_DIR="/mnt/fast/nobackup/scratch4weeks/ma04503/Sedimark/Api_outputs"
REQUEST_FILE="/mnt/fast/nobackup/scratch4weeks/ma04503/Sedimark/SEDIMARK_OFFERING/api/request.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Testing API at ${BASE_URL}"

# Check health
echo "Checking API health..."
curl -s "${BASE_URL}/health" | jq '.' 2>/dev/null

# Test with your specific input
echo "Sending request..."
USER_INPUT="Smart-lock entry logs from 120 green house plant in location: islamabad recorded in real time between March 1, 2023 and April 26, 2025"

# Use jq to merge if request.json exists
if [ -f "$REQUEST_FILE" ]; then
    curl -X POST "${BASE_URL}/test" \
         -H "Content-Type: application/json" \
         -d "$(jq --arg input "$USER_INPUT" '. + {user_input: $input}' "$REQUEST_FILE")" \
         -o "${OUTPUT_DIR}/output_${TIMESTAMP}.json" \
         -w "\nHTTP Status: %{http_code}\n"
else
    # Fallback without request.json
    curl -X POST "${BASE_URL}/test" \
         -H "Content-Type: application/json" \
         -d "{\"prompt\": \"Generate SEDIMARK offering\", \"user_input\": \"$USER_INPUT\"}" \
         -o "${OUTPUT_DIR}/output_${TIMESTAMP}.json" \
         -w "\nHTTP Status: %{http_code}\n"
fi

echo "Output saved to: ${OUTPUT_DIR}/output_${TIMESTAMP}.json"