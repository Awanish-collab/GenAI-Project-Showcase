from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test_output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="no",
)

print("✅ TrainingArguments initialized correctly!")
