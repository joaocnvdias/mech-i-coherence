import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left") #choose where padding will be applioed
tokenizer.pad_token_id = tokenizer.eos_token_id #required in llama because no padding token is defined
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

myinput=[
    [{"role": "user", "content": "1 + 1 = "}],
    [{"role": "user", "content": "Introduce C++ in one short sentence less than 10 words."}],
    [{"role": "user", "content": "Who was the first president of the United States? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the capital of France ? Answer in less than 10 words."}],
    [{"role": "user", "content": "Why is the sky blue ? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the meaning of life? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the best way to learn a new language? Answer in less than 10 words."}],
    [{"role": "user", "content": "When is the best time to plant a tree? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the best way to cook an egg? Answer in less than 10 words."}],
    [{"role": "user", "content": "Which is the best programming language? Answer in less than 10 words."}]
]

#convert a list of dictionaries with "role" and "content" keys to a list of token ids
texts = tokenizer.apply_chat_template(myinput, add_generation_prompt=True, tokenize=False) #prompt-adds token when the model should generate; tokenize- if we should tokenize the output, rn will be a string
inputs = tokenizer(texts, padding="longest", return_tensors="pt") #transform into pt (pytorch) tensors; pad to the longest sequence in the batch
inputs = {key: val.cuda() for key, val in inputs.items()} #move inputs into cuda
temp_texts=tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True) #way to debug inputs

start_time = time.time()
gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=512, 
    pad_token_id=tokenizer.eos_token_id, 
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)
print(f"Time: {time.time()-start_time}")

gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)] #remove the prompt text
print(gen_text)

