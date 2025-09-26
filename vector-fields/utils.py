import torch
import json
import spacy
from datasets import load_dataset

from EnergyComputations import energy_pipeline
from LLMfunctions import inference_activations

class ClauseSeparator:
    def __init__(self, size = 'small'):
        if size == 'small':
            self.nlp = spacy.load("en_core_web_sm")
        elif size == 'large':
            self.nlp = spacy.load("en_core_web_trf")
        else:
            print("Choose an appropriate spaCy model. Ex install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clause_split(self, text):
        """
        Find clause boundaries based on conjunctions and punctuation.    
        Note we don't need to add periods to the rules because we iterate through
        the sentences in the documents. 
        """
        if not self.nlp:
            return None
        
        doc = self.nlp(text)
        clauses = []
        
        for sentence in doc.sents:
            
            sent_clauses = []
            current_tokens = []
            
            for token in sentence:
                current_tokens.append(token.text)

                #check for what decides a new clause
                if (token.dep_ in ['cc', 'mark'] or  #coordinating/subordinating conjunctions
                    token.text in [',', ';', ':'] or
                    token.pos_ == 'SCONJ'):  #subordinating conjunction
                    
                    if len(current_tokens) > 1:  #dont create single-word clauses
                        clause_text = ' '.join(current_tokens[:-1]).strip() #everything before curent token - that will go to the next clause
                        if clause_text:
                            sent_clauses.append(clause_text)
                        current_tokens = [token.text] if token.text not in [',', ';', ':'] else [] #boundary token if not punctuation

            #sentence over, add remaining tokens of sentence as final clause
            if current_tokens:
                clause_text = ' '.join(current_tokens).strip()
                if clause_text:
                    sent_clauses.append(clause_text)
    
            #safeguard
            if not sent_clauses:
                sent_clauses.append(sentence.text.strip())
                
            clauses.extend(sent_clauses)
        
        return clauses

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

def load_tiny_stories():
    ds = load_dataset("roneneldan/TinyStories")
    tiny_train = ds['train']
    tiny_val = ds['validation']
    return tiny_train['text'], tiny_val['text']
    
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