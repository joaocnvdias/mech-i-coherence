from LLMfunctions import * 
from utils import *

def main():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    prompt_type = 'generation' #or 'completion'
    prompt_topic='inc1_newgen'
    nr_gens = 5
    prompt = load_prompts(prompt_type, prompt_topic)
    tokenizer, model, device, terminators = load_AutoModel(model_id, cuda_id=0)
    model_inputs = prepare_llama_prompt(tokenizer, prompt, device)
    gen_ids = llama_gen(model, model_inputs, tokenizer, terminators, num_generations=nr_gens)

    prompt_text = tokenizer.decode(model_inputs['input_ids'][0], skip_special_tokens=True)
    decoded_gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    decoded_stories = [decoded_generation[len(prompt_text):] for decoded_generation in decoded_gens] 
    
    for i in range(len(decoded_stories)):
        print(f'\nGENERATION NUMBER {i+1}:\n {decoded_stories[i]}\n')


if __name__ == '__main__':
    main()
