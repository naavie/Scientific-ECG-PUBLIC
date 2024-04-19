def calc_metrics(image_embeddings, captions, class_embeddings, class_names):
    similarity = nxn_cos_sim(image_embeddings, class_embeddings, dim=1)
    predictions_ids = similarity.argmax(dim=1)
    predictions = [class_names[idx] for idx in predictions_ids]
    tps = [prediction in caption for prediction, caption in zip(predictions, captions)]
    accuracy = np.mean(tps)

    results = dict()
    results['accuracy'] = accuracy

    similarity = similarity.detach().cpu().numpy()
    for i, name in enumerate(class_names):

        true = np.array([name in caption for caption in captions]).astype('int32')

        if true.std() > 0:
            results[f'{name}_rocauc'] = roc_auc_score(true, similarity[:, i])
            results[f'{name}_prauc'] = average_precision_score(true, similarity[:, i])
        else:
            results[f'{name}_rocauc'] = None
            results[f'{name}_prauc'] = None

    return results

def calc_accuracy(image_embeddings, captions, class_embeddings, class_names):
    similarity = nxn_cos_sim(image_embeddings, class_embeddings, dim=1)
    predictions_ids = similarity.argmax(dim=1)
    predictions = [class_names[idx] for idx in predictions_ids]
    tps = [prediction in caption for prediction, caption in zip(predictions, captions)]
    accuracy = np.mean(tps)
    return accuracy

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text