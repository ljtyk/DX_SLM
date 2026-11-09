from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, PreTrainedTokenizerFast
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "aligned-model"  # Folder from save_pretrained()

tokenizer_2 = Tokenizer.from_file("bpe_tokenizer.json")

# Wrap it
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_2,
    pad_token="<PAD>",
    bos_token="<BOS>",
    eos_token="<EOS>",
)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

while True:
    try:
        prompt = input("\nHuman: ")
        if prompt.lower() in {"exit", "quit"}:
            break

        full_prompt = f"Human: {prompt}\n\nAssistant:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=512,
                    do_sample=True,
                    temperature=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    top_k=50,
                    top_p=0.95,
                    #pad_token_id=tokenizer.eos_token_id
                )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        print(f"Assistant: {response}")

    except KeyboardInterrupt:
        break
