import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

def choose_device(device):
    if device==0:
        return 'cuda:0'
    elif device==1:
        return 'cuda:1'
    elif device ==-1:
        return 'cpu'
    else:
        raise Exception('Return 0 or 1 for GPUs or -1 for CPU')

def load_device(cuda_id):
    cuda = choose_device(cuda_id)
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    return device

def load_gpt2XL(cuda_id):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", output_hidden_states=True)
    model.eval()
    device = load_device(cuda_id)
    model = model.to(device)

    return tokenizer, model, device

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
    return [layer[0] for layer in full_outputs.hidden_states[1:]] #list with pt tensor of activations in each element

def load_AutoModel(model_id,cuda_id):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left") 
    tokenizer.pad_token_id = tokenizer.eos_token_id #required in llama because no padding token is defined
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    device = load_device(cuda_id)
    model = model.to(device)

    return tokenizer, model, device, terminators

def prepare_llama_prompt(tokenizer, prompt, device):
    text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) 
    inputs = tokenizer(text, padding="longest", return_tensors="pt") #transform into pt tensors
    inputs = {key: val.to(device) for key, val in inputs.items()} #move inputs into cuda
    return inputs

def llama_gen(model, inputs, tokenizer, terminators, num_generations):
    generations = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        num_return_sequences=num_generations  
    )

    return generations
    
def get_rephrased_clauses(narrative_part: str):
    client = OpenAI()
    
    system_prompt = "You are helping in scientific analysis, so please be precise."
    
    user_prompt = (
        "You will be given a part of a narrative, segmented into linguistic clauses and numbered from 1 to N.\n"
        "Your task is to generate a paraphrase of this part of the narrative, using different wording (lexical diversity) "
        "and phrasing (syntactic diversity), but keeping the meaning essentially the same.\n"
        "You should keep the numbering of the clauses in the paraphrase.\n\n"
        f"Part to paraphrase: ''' {narrative_part} '''"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cost-efficient, adjust as needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()