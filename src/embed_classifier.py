import pandas as pd
import numpy as np
import hashlib
import logging

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from utils import hash_string
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MilvusClassifier:
    def __init__(
        self,
        source_data: str,
        model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
        text_col: str = "text",
        label_col: str = "cluster_label",
        milvus_db: str = "milvus.db",
        milvus_collection: str = "milvus_demo",
        metric_type: str = "COSINE",
        index_type: str = "FLAT",
        **kwargs,
    ):
        """
        Initialize the MilvusClassifier.

        :param source_data: Path to the source CSV file
        :param model_name: Name of the SentenceTransformer model to use
        :param text_col: Name of the column containing text data
        :param label_col: Name of the column containing labels
        :param milvus_db: Milvus database connection string
        :param milvus_collection: Name of the Milvus collection to use
        :param metric_type: Type of distance metric to use
        :param index_type: Type of index to use
        :param kwargs: Additional keyword arguments for Milvus collection creation
        """
        logging.info("Initializing MilvusClassifier")

        self.source_data = source_data
        self.model_name = model_name
        self.label_col = label_col
        self.text_col = text_col
        self.metric_type = metric_type
        self.index_type = index_type

        self.collection_name = milvus_collection
        self.client = MilvusClient(milvus_db)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        self.schema = self._setup_schema(self.model.get_sentence_embedding_dimension())
        self.kwargs = kwargs
        if self.collection_name not in self.client.list_collections():
            self.build_milvus_index()
        else:
            self.update_index()

    def _setup_schema(self, dim=1024):
        logging.info(f"Setting up schema with dimension {dim}")
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(
                    name="cluster_avg_sim",
                    dtype=DataType.FLOAT,
                    description="Average cluster cosine similarity",
                ),
            ],
            enable_dynamic_field=True,
        )
        return schema

    def _setup_index(self, metric_type: str, index_type: str):
        logging.info(
            f"Setting up index with metric type {metric_type} and index type {index_type}"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type=metric_type,
        )
        return index_params

    def calculate_avg_cosine_similarity(
        self, df: pd.DataFrame, embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Calculate average cosine similarity for each cluster."""
        df = df.copy()
        assert df.shape[0] == embeddings.shape[0]
        df["embeddings"] = [np.array(embedding) for embedding in embeddings]

        assert self.label_col in df.columns, "cluster label column missing"

        avg_cosine_sims = (
            df.groupby(self.label_col)["embeddings"]
            .apply(np.stack)
            .apply(cosine_similarity)
            .apply(np.mean)
        ).to_dict()

        return avg_cosine_sims

    def build_milvus_index(self):
        """Build the Milvus index from the source data."""
        logging.info("Building Milvus index")
        df = pd.read_csv(self.source_data)
        assert df[self.label_col].notna().all(), "All labels must be non-null"

        df["hashed"] = df[self.text_col].apply(lambda x: hash_string(x))

        logging.info(f"Loaded source data from {self.source_data}")
        text_samples = df[self.text_col].tolist()
        embeddings = self.model.encode(text_samples)
        logging.info("Encoded texts to embeddings")

        avg_cosine_sims = self.calculate_avg_cosine_similarity(df, embeddings)
        self.index_params = self._setup_index(self.metric_type, self.index_type)

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=embeddings.shape[1],
            schema=self.schema,
            index_params=self.index_params,
            **self.kwargs,
        )
        logging.info(f"Created collection {self.collection_name}")

        data = self.format_insert(
            text_samples, df[self.label_col].tolist(), embeddings, avg_cosine_sims
        )
        self.client.insert(self.collection_name, data)
        df.to_csv(self.source_data, index=False)
        logging.info("Inserted data into Milvus collection")

    def _should_update_index(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check if the index needs updating and return IDs to delete and add."""
        logging.info("Checking if index update is needed")
        existing_ids = self.client.query(
            collection_name=self.collection_name,
            filter="pk != 0",
            output_fields=["id"],
        )
        existing_ids = {item["id"] for item in existing_ids}

        source_ids = set(df["hashed"].unique().tolist())
        ids_to_remove = list(existing_ids - source_ids)
        ids_to_add = list(source_ids - existing_ids)

        if ids_to_remove or ids_to_add:
            logging.info(
                f"Index needs update: {len(ids_to_remove)} IDs to remove, {len(ids_to_add)} IDs to add"
            )
        else:
            logging.info("No index update needed")

        return {
            "delete": ids_to_remove,
            "add": df[df["hashed"].isin(ids_to_add)].index.to_list(),
        }

    def update_records_cluster(self, clusters: List[str]):
        """Update cluster average similarities for given clusters."""
        res = self.client.query(
            collection_name=self.collection_name,
            filter=f"label in {clusters}",
            output_fields=["*"],
        )
        vectors = np.array([item["vector"] for item in res])
        clusters = [item["label"] for item in res]

        # FIX ME - remove this pandas dependency
        clusters = pd.DataFrame(
            data={
                self.label_col: clusters,
            }
        )

        clusters_sim = self.calculate_avg_cosine_similarity(clusters, vectors)
        for r in res:
            r["cluster_avg_sim"] = clusters_sim[r["label"]]

        self.client.upsert(collection_name=self.collection_name, data=res)

    def delete_by_ids(self, ids: List[str]):
        """Delete records from Milvus by their IDs."""
        if not ids:
            return

        result = self.client.delete(
            collection_name=self.collection_name,
            ids=ids,
        )
        logging.info(f"Deleted {len(result)} records from collection")

    @staticmethod
    def format_insert(texts, labels, embeddings, avg_cosine_sims):
        """Format data for insertion into Milvus."""
        data = [
            {
                "id": hashlib.sha256(str.encode(text)).hexdigest(),
                "label": label,
                "vector": embedding,
                "cluster_avg_sim": avg_cosine_sims.get(label, 0.0),
            }
            for label, embedding, text in zip(labels, embeddings, texts)
        ]
        return data

    def update_index(self):
        """Update the Milvus index with any changes in the source data."""
        df = pd.read_csv(self.source_data)
        df["hashed"] = df[self.text_col].apply(lambda x: hash_string(x))

        to_delete_add_files = self._should_update_index(df)
        clusters_to_recalc = set()

        if to_delete_add_files["delete"] or to_delete_add_files["add"]:
            if to_delete_add_files["delete"]:
                self.delete_by_ids(to_delete_add_files["delete"])
                clusters_to_recalc.update(
                    df[df["hashed"].isin(to_delete_add_files["delete"])][self.label_col].unique()
                )

            if to_delete_add_files["add"]:
                texts_labels = df.iloc[to_delete_add_files["add"]][
                    [self.text_col, self.label_col]
                ].reset_index(drop=True)

                embeddings = self.model.encode(texts_labels[self.text_col])
                new_data = self.format_insert(
                    texts=texts_labels[self.text_col],
                    labels=texts_labels[self.label_col],
                    embeddings=embeddings,
                    avg_cosine_sims={},
                )

                self.client.insert(collection_name=self.collection_name, data=new_data)
                clusters_to_recalc.update(texts_labels[self.label_col])
                logging.info(f"Inserted {len(new_data)} new records into collection")

            self.update_records_cluster(list(clusters_to_recalc))
            logging.info(
                f"Updated clustering average similarities for {len(clusters_to_recalc)} clusters"
            )

        collection_count = self.client.get_collection_stats(collection_name=self.collection_name)[
            "row_count"
        ]
        df.to_csv(self.source_data, index=False)
        assert (
            collection_count == df.shape[0]
        ), f"Mismatch in record count: Milvus has {collection_count}, DataFrame has {df.shape[0]}"

    def predict(self, query_text: str, k=5, fields=["*"]) -> List[str]:
        """Predict the top k similar records for the given query text."""
        logging.info(f"Predicting top {k} results for query text")
        results = self.client.search(
            collection_name=self.collection_name,
            data=self.model.encode([query_text]).tolist(),
            limit=k,
            output_fields=fields,
        )
        logging.info("Prediction completed")
        return results[0]
