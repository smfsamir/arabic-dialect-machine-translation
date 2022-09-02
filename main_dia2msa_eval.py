import itertools
import re
import pandas as pd
import evaluate
def calculate_chrf():
    reference_predictions_fname = "results/msa_to_egy_predictions_baseline.tsv"
    with open(reference_predictions_fname, 'r') as reference_predictions_f:
        reference_predictions_frame = pd.read_csv(reference_predictions_f, sep='\t')
        predictions = reference_predictions_frame['reference_msa'].values
        # predictions = reference_predictions_frame['prediction'].values
        references = reference_predictions_frame['reference'].values
        references = [[reference] for reference in references]
        chrf = evaluate.load('chrf')
        results = chrf.compute(predictions = predictions, references = references)
        print(results)
    
calculate_chrf()
