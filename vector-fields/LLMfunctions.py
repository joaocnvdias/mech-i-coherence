import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from EnergyComputations import energy_pipeline

def load_gpt2XL(device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", output_hidden_states=True)
    model.eval()

    cuda = load_device(device)
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return tokenizer, model, device

def load_device(device):
    if device==0:
        return 'cuda:0'
    elif device==1:
        return 'cuda:1'
    elif device ==-1:
        return 'cpu'
    else:
        raise Exception('Return 0 or 1 for GPUs or -1 for CPU')

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
    #remove 1st tensor dimension so its 2D
    return [layer[0] for layer in full_outputs.hidden_states] #list with pt tensor of activations in each element

def energy_loop(generated_ids, model):
    energy_values = []
    for i in range(generated_ids.shape[0]):
        tensor = generated_ids[i:i+1] #reshape to 1xseq_length
        activations = inference_activations(model, tensor)
        energy_values.append(energy_pipeline(activations))
    
    return energy_values

def energy_to_json(prompt_sufix, generated_ids, energy_values):
    file_ids = 'checkpoints/ids'+prompt_sufix+'.pt'
    file_energy = 'checkpoints/energy'+prompt_sufix+'.json'
    
    torch.save(generated_ids, file_ids) #save ids as pt
    with open(file_energy, 'w') as f:
        json.dump(energy_values, f) #save values as json