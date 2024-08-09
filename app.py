from transformers import pipeline
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

generator = pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)




class TextRequest(BaseModel):
    text: str
    max_length: int = 50  # Optional: Max length of the generated text

@app.post("/generate/")
async def generate_text(request: TextRequest):
    try:
        # Generate text using the pipeline
        result = generator(request.text, max_length=request.max_length, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))