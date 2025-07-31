from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('./t5_recipe_model')
model = T5ForConditionalGeneration.from_pretrained('./t5_recipe_model')

def generate_recipe(query):
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    query = input("\n👤 You: ")
    if query.lower() in ["exit", "quit"]: break
    print("🤖 RecipeBot:", generate_recipe(query))
