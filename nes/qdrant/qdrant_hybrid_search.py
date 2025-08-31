from qdrant_client import QdrantClient, models
import logging
import os
from nes.qdrant.fastembed_functions import get_dense_model_vector_size, get_late_interaction_model_vector_size

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

        self.dense_dim = get_dense_model_vector_size(QDRANT_EMB_DENSE_MODEL_NAME)
        self.late_interaction_dim = get_late_interaction_model_vector_size(QDRANT_EMB_LATE_ITER_MODEL_NAME)

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

    def update_indexes(self):
        logging.info("Очікування завершення індексації...")
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    def add_point(self, doc_id, payload, dense_embedding, sparse_embedding, late_interaction_embedding):
        point = models.PointStruct(
            id=doc_id,
            vector={
                QDRANT_EMB_DENSE_VECTOR_NAME: dense_embedding,
                QDRANT_EMB_SPARSE_VECTOR_NAME: sparse_embedding,
                QDRANT_EMB_LATE_ITER_VECTOR_NAME: late_interaction_embedding,
            },
            payload=payload
        )

        self.add_points([point])

        return point

    def add_points(self, points):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=False
        )

    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # Щільний вектор
                QDRANT_EMB_DENSE_VECTOR_NAME: models.VectorParams(
                    size=self.dense_dim,
                    distance=models.Distance.COSINE,
                ),
                # Мульти-вектор (багато векторів на документ)
                QDRANT_EMB_LATE_ITER_VECTOR_NAME: models.VectorParams(
                    size=self.late_interaction_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    )
                ),
            },
            # Розріджений вектор
            sparse_vectors_config={
                QDRANT_EMB_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )

    def create_collection_if_not_exists(self, collection_name: str):
        if not self.client.get_collection(collection_name=collection_name):
            self.create_collection(collection_name)