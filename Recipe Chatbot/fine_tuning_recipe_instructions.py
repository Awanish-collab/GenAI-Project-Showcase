from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType
import json
from datasets import Dataset

# Load dataset
#dataset = load_dataset('json', data_files='./recipe_dataset.json')['train']
with open("recipe_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# PEFT configuration (LoRA)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8, lora_alpha=32, lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

def tokenize(batch):
    inputs = tokenizer(batch["instruction"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(tokenize)
collator = DataCollatorForSeq2Seq(tokenizer, model)

args = Seq2SeqTrainingArguments(
    output_dir="./recipe_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=1
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=collator
)

trainer.train()
