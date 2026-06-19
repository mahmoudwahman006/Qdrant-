## QDRANT : Intro in Vector stores focusing on retrical and metadata filtring : 

# 1- Imports 
from qdrant_client import QdrantClient
from qdrant_client.models import  VectorParams, Distance, PointStruct # for the size and dimes, for measure dims, for individual data points 
import shutil


# 2- remove the existing directory if it exists

shutil.rmtree('./qdrant_data', ignore_errors=True)  # Remove the existing directory if it exists

# 3- Create a Qdrant client instance

# client = QdrantClient(host="localhost", port=6333)  # Docker running Qdrant on localhost
client = QdrantClient(path='./qdrant_data')  # Local mode without docker, but only one process can access it at a time


# 4- Create a collection with vector size and distance metric
client.create_collection(
    collection_name="first_collection",
    vectors_config=VectorParams(
        size=2,
        distance=Distance.EUCLID   # theroy distance is to measure the distance between two points directly (Diagonal)
    )
)
"""
Comparison between distance metrics:
1- Euclidean (EUCLID) : Measures the straight-line distance between two points in a multi-dimensional space. 
                        It is sensitive to the scale of the data (magnitude) and can be affected by outliers.
                        Appropriate for low-dimensional spaces and when the magnitude of the vectors is important.
                        [Image Embeddings, Clustering, and Anomaly Detection]

2- Cosine (COSINE) : Measures the cosine of the angle between two vectors (ignoring magnitude. Range -1 to 1) 
                     Larger the range = more similar the vectors are. It is not sensitive to the magnitude of the vectors, 
                     Making it suitable for text data and high-dimensional spaces 
                     It focuses on the direction of the vectors rather than their length.
                     Appropriate for high-dimensional spaces and when the direction of the vectors is more important than their magnitude.
                     [Text Similarity, Document Retrieval, Search engines, and deduplication]
                     
3- Dot Product (DOT) : Measures the dot product of two vectors 
                       Which is the sum of the products of their corresponding elements (Larger = more similar)
                       Suitable for longer vectores, Affected by both the magnitude and direction of the vectors
                       Making it useful for certain applications like recommendation systems.
                       [Recommendation Systems, Collaborative Filtering, and Ranking]
"""

# 5- Insert points into the collection

client.upsert(
    collection_name="first_collection",
    wait=True,                                    ################ important 
    points=[
        PointStruct(id=1, vector=[0.0, 0.0], payload={"city": "New York", "country": "USA"}),
        PointStruct(id=2, vector=[1.0, 1.0], payload={"city": "Los Angeles", "country": "USA"}),
        PointStruct(id=3, vector=[2.0, 2.0], payload={"city": "London", "country": "UK"}),
        PointStruct(id=4, vector=[3.0, 3.0], payload={"city": "Paris", "country": "France"}),
        PointStruct(id=5, vector=[4.0, 4.0], payload={"city": "Berlin", "country": "Germany"}),
    ]


) 


# 6- Query points : 

results = client.query_points(
    collection_name="first_collection",
    query=[1.5, 1.5],
    limit=3,
    with_payload=True).points 


# 7- Print the results: 

print("Query Results: ", results) 
