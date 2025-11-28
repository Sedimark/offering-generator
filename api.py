import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import CONFIG
from model_loader import ModelLoader
from offering_generator import SEDIMARKOfferingGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="SEDIMARK Offering Generator",
    description="Generate SEDIMARK-compliant JSON-LD offerings",
    version="2.0.0"
)

# Security
auth_scheme = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """JWT verification (simplified)"""
    token = credentials.credentials
    try:
        import jwt
        payload = jwt.decode(token, CONFIG["jwt_secret"], algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    use_context: Optional[bool] = Field(False)
    context: Optional[Dict[str, Any]] = Field(None)
    use_schema: Optional[bool] = Field(False)
    max_new_tokens: Optional[int] = Field(4096, ge=1, le=8192)
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    num_beams: Optional[int] = Field(4, ge=1, le=8)

class GenerateResponse(BaseModel):
    offering: Dict[str, Any]
    metadata: Dict[str, Any]

# Global variables
model_loader = None
offering_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_loader, offering_generator

    try:
        logger.info("Loading model...")
        model_loader = ModelLoader()
        model_loader.load_model()
        offering_generator = SEDIMARKOfferingGenerator(model_loader)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SEDIMARK Offering Generator",
        "version": "2.0.0",
        "status": "ready" if offering_generator else "loading",
        "endpoints": {
            "generate": "/generate",
            "generate_simple": "/api/offerings/generate",
            "health": "/health"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_offering(request: GenerateRequest):
    """Generate SEDIMARK offering with detailed control"""
    if not offering_generator:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Extract generation parameters
        gen_params = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_beams": request.num_beams,
        }

        # Generate offering
        offering = offering_generator.generate_offering(
            prompt=request.prompt,
            use_context=request.use_context,
            context=request.context,
            use_schema=request.use_schema,
            **gen_params
        )

        generation_time = time.time() - start_time

        # Add metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "generation_time_seconds": round(generation_time, 2),
            "model": "SEDIMARK-Qwen2.5-3B",
            "use_context": request.use_context,
            "use_schema": request.use_schema,
            "generation_params": gen_params
        }

        return GenerateResponse(offering=offering, metadata=metadata)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/api/offerings/generate")
async def generate_simple(
    prompt: str = Query("Generate a SEDIMARK offering for IoT sensor data"),
    count: int = Query(1, ge=1, le=5),
    use_context: bool = Query(False),
    use_schema: bool = Query(False),
    save: bool = Query(True)
):
    """Simple offering generation endpoint"""
    if not offering_generator:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if count == 1:
            offering = offering_generator.generate_offering(
                prompt=prompt,
                use_context=use_context,
                use_schema=use_schema
            )

            offering["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "generation_time_seconds": round(time.time() - start_time, 2),
                "model": "SEDIMARK-Qwen2.5-3B"
            }

            # Save if requested
            if save:
                output_file = CONFIG["output_dir"] / f"offering_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(offering, f, indent=2)
                offering["_metadata"]["saved_to"] = str(output_file)

            return JSONResponse(content=offering, media_type="application/ld+json")

        else:
            # Multiple offerings
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
                "@context": offering_generator.schema_data.get("@context", {}) if offering_generator.schema_data else {},
                "_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generation_time_seconds": round(time.time() - start_time, 2),
                    "count": count,
                    "model": "SEDIMARK-Qwen2.5-3B"
                }
            }

            if save:
                output_file = CONFIG["output_dir"] / f"offerings_{count}x_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(combined_offering, f, indent=2)
                combined_offering["_metadata"]["saved_to"] = str(output_file)

            return JSONResponse(content=combined_offering, media_type="application/ld+json")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status_info = {
        "status": "healthy" if offering_generator else "unhealthy",
        "model_loaded": offering_generator is not None,
        "timestamp": datetime.now().isoformat()
    }

    if model_loader and model_loader.tokenizer:
        status_info.update({
            "vocab_size": len(model_loader.tokenizer),
            "device": str(model_loader.device)
        })

    return status_info

@app.get("/api/offerings/list")
async def list_offerings():
    """List generated offerings"""
    try:
        output_dir = Path(CONFIG["output_dir"])
        files = sorted(output_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

        offerings_list = []
        for file in files[:20]:  # Last 20 files
            stat = file.stat()
            offerings_list.append({
                "filename": file.name,
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return {
            "output_directory": str(output_dir),
            "total_files": len(files),
            "recent_offerings": offerings_list
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=CONFIG["server"]["host"],
        port=CONFIG["server"]["port"],
        workers=CONFIG["server"]["workers"],
        reload=False
    )
