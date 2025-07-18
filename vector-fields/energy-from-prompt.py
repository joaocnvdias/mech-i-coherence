from LLMfunctions import * 
from utils import *

def main():

    #variables
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    prompt_type = 'generation' #or 'completion'
    prompt_topic='inc_1'
    nr_gens = 200

    prompt = load_prompts(prompt_type, prompt_topic)
    tokenizer, model, device, terminators = load_AutoModel(model_id, cuda_id=0)
    model_inputs = prepare_llama_prompt(tokenizer, prompt, device)
    gen_ids = llama_gen(model, model_inputs, tokenizer, terminators, num_generations=nr_gens)
    energy_values = energy_loop_llama(model_inputs, nr_gens, gen_ids, model)
    energy_to_json(prompt_topic, gen_ids, energy_values)

if __name__ == '__main__': 
    main()