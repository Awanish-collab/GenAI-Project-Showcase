from datasets import load_dataset
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


# Load CSV dataset
dataset = load_dataset('csv', data_files='data/recipe_dataset_t5.csv')

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenization function
def tokenize(batch):
    input_enc = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=64)
    target_enc = tokenizer(batch['output'], padding='max_length', truncation=True, max_length=64)

    input_enc['labels'] = target_enc['input_ids']
    return input_enc

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Training arguments
training_args = TrainingArguments(
    output_dir='./t5_recipe_model',
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./t5_recipe_model")
tokenizer.save_pretrained("./t5_recipe_model")
