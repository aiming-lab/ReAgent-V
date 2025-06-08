from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever')

def text_to_vector(text, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def retrieve_documents_with_dynamic(documents, queries, threshold=0.4):
    if isinstance(queries, list):
        query_vectors = np.array([text_to_vector(query) for query in queries])
        average_query_vector = np.mean(query_vectors, axis=0)
        query_vector = average_query_vector / np.linalg.norm(average_query_vector)
        query_vector = query_vector.reshape(1, -1)
    else:
        query_vector = text_to_vector(queries)
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1)

    document_vectors = np.array([text_to_vector(doc) for doc in documents])
    document_vectors = document_vectors / np.linalg.norm(document_vectors, axis=1, keepdims=True)
    dimension = document_vectors.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(document_vectors)
    lims, D, I = index.range_search(query_vector, threshold)
    start = lims[0]
    end = lims[1]
    I = I[start:end]

    if len(I) == 0:
        top_documents = []
        idx = []
    else:
        idx = I.tolist()
        top_documents = [documents[i] for i in idx]

    return top_documents, idx
