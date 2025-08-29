from fastembed import TextEmbedding, LateInteractionTextEmbedding

def get_dense_model_vector_size(model: str):
    for emb_model in TextEmbedding.list_supported_models():
        if emb_model['model'] == model:
            return emb_model['dim']

    return None

def get_late_interaction_model_vector_size(model: str):
    for emb_model in LateInteractionTextEmbedding.list_supported_models():
        if emb_model['model'] == model:
            return emb_model['dim']

    return None