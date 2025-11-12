import krippendorff
import numpy as np
import pickle

def load_dataset(dataset_path):
    
    with open(dataset_path,'rb') as f:
        return pickle.load(f)

def krippendorffs_alpha(outputs,groundtruth):

    if len(outputs) != len(groundtruth):
        raise ValueError("outputs and groundtruth must be the same length")
    data = np.array([outputs, groundtruth])
    alpha = krippendorff.alpha(reliability_data=data)
    print(f"Krippendorff's alpha: {alpha}")
    return alpha

def evaluate_scores(pos_scores,neg_scores):
    draw = 0
    winners = []
    if len(pos_scores)!= len(neg_scores):
        raise Exception('Positive and negative scores dont have the same length')
    
    for i in range(len(pos_scores)):
        pos_value = int(pos_scores[i])
        neg_value = int(neg_scores[i])

        if pos_value > neg_value:
            winner = 0
        elif pos_value < neg_value:
            winner = 1
        elif pos_value == neg_value:
            winner = np.nan
            draw +=1
        winners.append(winner)

    percentage_draws = np.round((draw/len(pos_scores))*100, 2)
    print(f'Finished evaluting positives vs negatives, obtained a total of {draw} draws ({percentage_draws}% of total data)')
    return winners

def main():

    annotations = load_dataset('datasets/LMvLM_Annotations.pkl')
    annotations_int = [int(indv) for indv in annotations[0]]
    pos_scores = np.load('outputs/LMvLMpos.npy')
    neg_scores = np.load('outputs/LMvLMneg.npy')

    winners = evaluate_scores(list(pos_scores),list(neg_scores))
    
    kalpha = krippendorffs_alpha(winners, annotations_int)

if __name__ == '__main__':
    main()