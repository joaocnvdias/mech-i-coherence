import torch

def generate_multiple_completion(tokenizer, device, model, prompt, repetion_value=1.2, printing=False,num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    #generate continuation from the prompt
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            repetition_penalty=repetion_value, #without this gen is very repetitive 
            top_p=0.95,
            return_dict_in_generate=True,
            output_hidden_states=False,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
        )
    
    generated_ids = outputs.sequences

    if printing == True:
        for i, generation in enumerate(generated_ids):
            generated_text = tokenizer.decode(generation, skip_special_tokens=True)
            print(f'Generated text:\n {generated_text}\n')
    
    return generated_ids

def inference_activations(model, gen_ids):
    #pass the full generation through the model 
    with torch.no_grad():
        full_outputs = model(
            input_ids=gen_ids,
            output_hidden_states=True,
            return_dict=True
        )

    return [layer[0] for layer in full_outputs.hidden_states] #list with pt tensor of activations in each element