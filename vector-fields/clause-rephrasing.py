from utils import * 

train, test = load_tiny_stories()
rephrase_index = 0

clause_model = ClauseSeparator()
clauses_list = clause_model.clause_split(train[rephrase_index]) 
