import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import *
from LLMfunctions import get_rephrased_clauses

def worker(i, clauses_to_rephrase, model_name, temp):
    """Wrapper to call the API and return index + result for ordering."""
    result = get_rephrased_clauses(clauses_to_rephrase, model_name, temp)
    return i, result

def main():
    #load stories and define script variables
    train, test = load_tiny_stories()
    rephrase_index = 2
    nr_rephrasing = 5
    model_name = 'gpt-5-mini' #'gpt-o4-mini'
    temp = 1.5


    #prepare clauses
    clause_model = ClauseSeparator()
    clauses_list = clause_model.clause_split(train[rephrase_index])
    numbered_clauses = clause_model.number_clauses(clauses_list)

    #container for rephrasings
    rephrasings = [None] * nr_rephrasing  #pre-allocate to preserve order

    #parallel execution
    max_workers = 25
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i, numbered_clauses, model_name, temp) for i in range(nr_rephrasing)]
        for future in as_completed(futures):
            i, result = future.result()
            rephrasings[i] = result  #preserve original order

    #save to json
    file_path = "rephrasings/TEST.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rephrasings, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rephrasings)} rephrasings to {file_path}")

if __name__ == '__main__':
    main()