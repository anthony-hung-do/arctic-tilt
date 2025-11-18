import re
from collections import Counter
import string
from anls import anls_score
import evaluate

squad_metric = evaluate.load("squad")

def compute_docqa_metrics(predictions, references):
    """
    Tính toán metrics cho Document QA với multiple references support
    predictions: List[str]
    references: List[List[str]]
    """
    # ANLS scores
    anls_scores = []
    for pred, refs in zip(predictions, references):
        score = anls_score(prediction=pred, gold_labels=refs, threshold=0.5)
        anls_scores.append(score)

    # Prepare for SQuAD metric (F1 score and EM)
    squad_predictions = []
    squad_references = []

    for i, (pred, ref_list) in enumerate(zip(predictions, references)):
        sample_id = str(i)

        squad_predictions.append({
            "id": sample_id,
            "prediction_text": pred
        })

        squad_references.append({
            "id": sample_id,
            "answers": {
                "text": ref_list,
                "answer_start": [0] * len(ref_list)  # Dummy answer_start positions
            }
        })

    squad_results = squad_metric.compute(
        predictions=squad_predictions,
        references=squad_references
    )

    return {
        'exact_match': squad_results['exact_match'],
        'f1': squad_results['f1'],
        "anls": sum(anls_scores) / len(anls_scores)
    }