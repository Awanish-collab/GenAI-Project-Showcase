from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

# Load the fine-tuned model
model_path = "recipe_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

@app.post("/generate/")
def generate_text(query: PromptInput):
    formatted_prompt = f"Write a complete recipe using: {query.prompt}"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
