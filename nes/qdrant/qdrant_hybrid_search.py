from qdrant_client import QdrantClient, models
import logging
import os

# Щільні (Dense) вектори - для семантичного розуміння.
QDRANT_EMB_DENSE_MODEL_NAME = os.environ.get("QDRANT_EMB_DENSE_MODEL_NAME")
# Розріджені (Sparse) вектори - для пошуку за ключовими словами.
QDRANT_EMB_SPARSE_MODEL_NAME = os.environ.get("QDRANT_EMB_SPARSE_MODEL_NAME")
# Мульти-вектори  - для точного зіставлення на рівні токенів.
QDRANT_EMB_LATE_ITER_MODEL_NAME = os.environ.get("QDRANT_EMB_LATE_ITER_MODEL_NAME")

""""""
QDRANT_EMB_DENSE_VECTOR_NAME = QDRANT_EMB_DENSE_MODEL_NAME.replace("/", "-").replace(":", "-")
QDRANT_EMB_SPARSE_VECTOR_NAME = QDRANT_EMB_SPARSE_MODEL_NAME.replace("/", "-").replace(":", "-")
QDRANT_EMB_LATE_ITER_VECTOR_NAME = QDRANT_EMB_LATE_ITER_MODEL_NAME.replace("/", "-").replace(":", "-")

def log_fetched_scored_points(data):
    for point in data.points:
        logging.info(f"ID: {point.id}; Score: {point.score}")
        logging.info(f"{point.payload['title']}")

def format_docs_from_qdrant(docs) -> str:
    # Додаємо вивід джерел для відладки
    for doc in docs:
        logging.info(f"Qdrant Source: {doc}")
    return "\n\n".join(f"Зі статті: {doc.metadata['source']}" for doc in docs)

class QdrantHybridSearchClient:
    """

    """
    client: QdrantClient
    
    collection_name: str
    
    internal_limit: int = 20

    results_limit: int = 10

    is_with_payload: bool = False
    
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def rrf_prefetch(self, dense_query_vector, sparse_query_vector, ):
        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=QDRANT_EMB_DENSE_VECTOR_NAME,
                limit=self.internal_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using=QDRANT_EMB_SPARSE_VECTOR_NAME,
                limit=self.internal_limit,
            ),
        ]
        return self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )
    
    def full_rrf_prefetech(self, dense_query_vector, sparse_query_vector, late_query_vector):
        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=QDRANT_EMB_DENSE_VECTOR_NAME,
                limit=self.internal_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using=QDRANT_EMB_SPARSE_VECTOR_NAME,
                limit=self.internal_limit,
            ),
            models.Prefetch(
                query=late_query_vector,
                using=QDRANT_EMB_LATE_ITER_VECTOR_NAME,
                limit=self.internal_limit,
            ),
        ]
    
        return self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )
    
    def reranking_prefetech(self, dense_query_vector, sparse_query_vector, late_query_vector):
        """ Reranking with late interaction model """
        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=QDRANT_EMB_DENSE_VECTOR_NAME,
                limit=self.internal_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using=QDRANT_EMB_SPARSE_VECTOR_NAME,
                limit=self.internal_limit,
            ),
        ]
        return self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=late_query_vector,
            using=QDRANT_EMB_LATE_ITER_VECTOR_NAME,
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )
    
    def multistep_prefetch(self, dense_query_vector, sparse_query_vector, late_query_vector):
        """ Multistep retrieval process """
        return self.client.query_points(
            self.collection_name,
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=dense_query_vector,
                            using=QDRANT_EMB_DENSE_VECTOR_NAME,
                            limit=self.internal_limit * 2,
                        )
                    ],
                    query=models.SparseVector(**sparse_query_vector.as_object()),
                    using=QDRANT_EMB_SPARSE_VECTOR_NAME,
                    limit=self.internal_limit,
                ),
            ],
            query=late_query_vector,
            using=QDRANT_EMB_LATE_ITER_VECTOR_NAME,
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )

    def dense_fetch(self, dense_query_vector):
        return self.client.query_points(
            self.collection_name,
            query=dense_query_vector,
            using=QDRANT_EMB_DENSE_VECTOR_NAME,
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )

    def sparse_fetch(self, sparse_query_vector):
        return self.client.query_points(
            self.collection_name,
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using=QDRANT_EMB_SPARSE_VECTOR_NAME,
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )

    def late_interaction_fetch(self, late_interaction_query_vector):
        return self.client.query_points(
            self.collection_name,
            query=late_interaction_query_vector,
            using=QDRANT_EMB_LATE_ITER_VECTOR_NAME,
            with_payload=self.is_with_payload,
            limit=self.results_limit,
        )