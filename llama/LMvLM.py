import transformers
import torch 
import pickle    
import random
import json
import time

def import_dataset(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def shuffle_dataset(dataset):
    random.seed(10)
    subset_idx = random.sample(range(len(dataset)), len(dataset))
    positives = [dataset[i]['pos'] for i in subset_idx]
    negatives = [dataset[i]['neg'] for i in subset_idx]
    return positives,negatives, subset_idx #shuffled dataset

def save_scores(scores, id_order, filename='outputs/scores.json'): #need to fix bug
    scores_flatten = [score for sublist in scores for score in sublist]
    data = dict(zip(id_order[0:len(scores_flatten)],scores_flatten)) #handle the case if not all texts are rated
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

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
    lmvlm = import_dataset('datasets/LMvLM.pkl')
    positives, negatives, id_order = shuffle_dataset(lmvlm)
    scores_p = call_llama(positives)
    scores_n = call_llama(negatives)
    save_scores(scores_p, id_order, 'outputs/scores_p.json')
    save_scores(scores_n, id_order, 'outputs/scores_n.json')

if __name__ == '__main__':
    main()