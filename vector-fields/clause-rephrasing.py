from utils import * 
from LLMfunctions import get_rephrased_clauses

train, test = load_tiny_stories()
rephrase_index = 0

clause_model = ClauseSeparator()
clauses_list = clause_model.clause_split(train[rephrase_index]) 
numbered_clauses = clause_model.number_clauses(clauses_list)

rephrased = get_rephrased_clauses(numbered_clauses)
