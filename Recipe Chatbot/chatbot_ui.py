# Recipe Chatbot/chatbot_ui.py

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load fine-tuned model
model_path = "recipe_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

# Response generation function
def generate_response(prompt):
    formatted_prompt = f"Write a complete recipe using: {prompt}"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI setup
examples = [
    "tomatoes and basil",
    "rice and peas",
    "eggs and spinach",
    "paneer and capsicum",
    "banana and oats"
]

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Ingredients or Prompt", placeholder="e.g. Make a recipe with tomatoes and basil"),
    outputs=gr.Textbox(label="Generated Recipe"),
    title="🧠 Recipe Maker Chatbot",
    description="Enter ingredients or a prompt and get a complete recipe. Model is trained on custom dataset.",
    examples=examples
)

demo.launch()