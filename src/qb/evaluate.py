import tqdm
import numpy as np


def compute_accuracy(predictor, instances):
    vocab = predictor._model.vocab
    batch_size = 32
    idx = 0
    preds = []
    progress = tqdm.tqdm(total=len(instances))
    while idx < len(instances):
        batch = instances[idx:idx + batch_size]
        preds.extend(predictor.predict_batch_instance(batch))
        idx += batch_size
        progress.update(idx)
    pred_pages = []
    correct = []
    for example, p in zip(instances, preds):
        predicted_page = vocab.get_token_from_index(
            p['preds'],
            namespace='page_labels'
        )
        correct.append(example['metadata']['page'] == predicted_page)
        pred_pages.append(predicted_page)
    return np.mean(correct)