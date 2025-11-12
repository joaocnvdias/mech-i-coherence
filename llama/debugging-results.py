import time
import transformers
import torch
import pickle
import numpy as np 

def load_dataset(dataset_path):
    with open(dataset_path,'rb') as f:
        return pickle.load(f)


def parse_dataset(dataset):
    positives = [instance['pos'] for instance in dataset]
    negatives = [instance['neg'] for instance in dataset]
    return positives,negatives

def create_prompt(story):
    return [{"role": "user", 
             "content": "Below is a text extract. Your task is to analyze the extract and assign a coherence score between 0 and 5 inclusive, where:\n\n"
            "0: The text is completely incoherent and lacks any logical connection.\n"
            "1: The text has some minor connections, but overall it is disjointed and hard to follow.\n"
            "2: The text has some coherence, but it is still difficult to understand due to unclear relationships between ideas.\n"
            "3: The text is moderately coherent, with some clear connections between ideas, but may lack depth or clarity.\n"
            "4: The text is highly coherent, with clear and logical connections between ideas, making it easy to follow.\n"
            "5: The text is extremely coherent, with a clear and concise structure, making it effortless to understand.\n\n"
            "You will provide a score ONLY. Do NOT also provide an explanation.\n"
            f"The extract: {story}\n"
            "After examining the extract, the coherence score between 0 and 5 inclusive is:"}]

def create_prompt_batch(batch):
    input_text = []
    for text in batch:
        input_text.append(create_prompt(text))
    return input_text

def call_llama(dataset, batch_size = 16):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side = "left") #choose where padding will be applioed
    tokenizer.pad_token_id = tokenizer.eos_token_id #required in llama because no padding token is defined
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    start_time = time.time()
    full_scores = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size] 
        input_text = create_prompt_batch(batch)

        texts = tokenizer.apply_chat_template(input_text, add_generation_prompt=True, tokenize=False) #prompt-adds token when the model should generate; tokenize- if we should tokenize the output, rn will be a string
        inputs = tokenizer(texts, padding="longest", return_tensors="pt") #transform into pt (pytorch) tensors; pad to the longest sequence in the batch
        inputs = {key: val.cuda() for key, val in inputs.items()} #move inputs into cuda
        temp_texts=tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True) #way to debug inputs

        gen_tokens = model.generate(
            **inputs, 
            max_new_tokens=10, 
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )

        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        gen_scores = [int(i[len(temp_texts[idx]):]) for idx, i in enumerate(gen_text)] #remove the prompt text
        print(f'\n Rated {i+batch_size} texts out of {len(dataset)} - {(i+batch_size)/len(dataset):.2%}')
        print(f'We obtained the following scores: {gen_scores}')
        full_scores.append(gen_scores)
      
    print(f"Time Elapsed: {time.time()-start_time}")

    return full_scores

def main():
    LMvLM = load_dataset('datasets/LMvLM.pkl')
    pos,neg = parse_dataset(LMvLM)

    pos_scores = call_llama(pos)
    neg_scores = call_llama(neg)

    np.save('outputs/LMvLMpos.npy', np.array([score for batch in pos_scores for score in batch]))
    np.save('outputs/LMvLMneg.npy', np.array([score for batch in neg_scores for score in batch]))

if __name__ == '__main__': 
    main()