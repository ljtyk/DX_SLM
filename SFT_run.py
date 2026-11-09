from transformers import pipeline, AutoTokenizer

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Make sure pad_token is set

pipe = pipeline("text-generation", model="./sft-neo-qlora-final", tokenizer=tokenizer, device=0)

prompt = "### Instruction:\nWhat's 4+5?\n\n### Response:\n"
out = pipe(prompt, max_new_tokens=100, do_sample=False, temperature=0.7)
print(out[0]["generated_text"])