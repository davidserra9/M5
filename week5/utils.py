import numpy as np
def aggregate_text_embedding(sentence, embedding_model):
    """
    Aggregate the text embedding of a sentence using the embedding model.
    """
    return np.mean(embedding_model.wv[sentence], axis=0)