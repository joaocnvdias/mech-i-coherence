from LLMfunctions import * 

def main():

    #load prompt from .txt
    prompt_topic = 'spaceship'
    prompt_sufix = '_' + prompt_topic
    with open('prompts/'+prompt_topic+'.txt') as file:
        prompt = file.read()
    
    tokenizer, model, device = load_gpt2XL(1) #0 for cuda:0 and 1 for cuda:1
    generated_ids = generate_multiple_completion(tokenizer, device, model, prompt,repetion_value=1.2, printing=False, num_return_sequences=200)
    energy_values = energy_loop(generated_ids, model)
    energy_to_json(prompt_sufix, generated_ids, energy_values)

if __name__ == '__main__': 
    main()