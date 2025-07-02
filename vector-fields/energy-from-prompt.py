from LLMfunctions import * 

def main():

    # prompt = "A brave knight stood before the cave, sword drawn. The dragon's breath steamed in the cold air as"
    #prompt = "In a kingdom where dragons are the last remnants of ancient magic, a young girl discovers a dragon egg buried beneath her village well. She decides to..."
    prompt = "The spaceship Hyperion lost contact with Earth 200 years ago. When it finally returned, its crew hadn't aged a day. According to the captain, they had only been gone for..."
    prompt_sufix = '_spaceship'

    tokenizer, model, device = load_gpt2XL(1) #0 for cuda:0 and 1 for cuda:1
    generated_ids = generate_multiple_completion(tokenizer, device, model, prompt,repetion_value=1.2, printing=False, num_return_sequences=200)
    energy_values = energy_loop(generated_ids, model)
    energy_to_json(prompt_sufix, generated_ids, energy_values)

if __name__ == '__main__': 
    main()