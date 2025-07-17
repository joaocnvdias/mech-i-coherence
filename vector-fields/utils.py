import torch
import json
from EnergyComputations import energy_pipeline
from LLMfunctions import inference_activations

def load_prompts(prompt_type, prompt_topic):
    #load prompt from .txt
    if prompt_type == 'completion':
        with open('prompts-comp/'+prompt_topic+'.txt') as file:
            prompt = file.read()
    elif prompt_type == 'generation':
        with open('prompts-gen/'+prompt_topic+'.txt') as file:
            prompt = file.read()
        prompt = json.loads(prompt, strict=False)
    return prompt
    
def energy_loop(generated_ids, model):
    energy_values = []
    for i in range(generated_ids.shape[0]):
        tensor = generated_ids[i:i+1] #reshape to 1xseq_length
        activations = inference_activations(model, tensor)
        energy_values.append(energy_pipeline(activations))
    
    return energy_values

def energy_loop_llama(model_inputs, num_generations, generations, model):
    tensor_size_prompt = model_inputs['input_ids'].shape[1] #obtain prompt token size
    energy_values = []
    for i in range(num_generations):
        tensor = generations[i,tensor_size_prompt:].unsqueeze(0) #remove prompt tokens from generation
        activations = inference_activations(model,tensor)
        energy_values.append(energy_pipeline(activations))

    return energy_values

def energy_to_json(prompt_topic, generated_ids, energy_values):
    file_ids = 'checkpoints/ids'+'_'+prompt_topic+'.pt'
    file_energy = 'checkpoints/energy'+'_'+ prompt_topic+'.json'
    
    torch.save(generated_ids, file_ids) #save ids as pt
    with open(file_energy, 'w') as f:
        json.dump(energy_values, f) #save values as json