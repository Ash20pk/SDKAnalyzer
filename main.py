from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.analyzer import analyze_sdk
import logging

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SDKRequest(BaseModel):
    package_name: str
    version: str = 'latest'

@app.post("/analyze")
async def analyze(sdk_request: SDKRequest):
    result = analyze_sdk(sdk_request.package_name, sdk_request.version)
    if "error" in result:
        logging.error(f"Error analyzing SDK: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])
    return {"result": result}

@app.get("/")
async def root():
    return {"message": "Welcome to the SDK Analyzer API"}