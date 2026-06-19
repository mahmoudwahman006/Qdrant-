
# example for using cohere embbeding in qdrant and the retrival : 
# note I use a old version of cohere 5.0.0 so it's diffrent from the current docuentation


# 1- Imports 
import os

from qdrant_client import QdrantClient
from qdrant_client.models import  VectorParams, Distance, PointStruct           # for the size and dimes, for measure dims, for individual data points 
from qdrant_client.models import Filter, FieldCondition, MatchValue             # for filtering the data points based on conditions (hard filter not ai based)

import shutil                                                                   # for removing the existing directory if it exists
import uuid                                                                     # for generating unique identifiers for each point

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  

# Initialize the Cohere client
import cohere 
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))  # Replace with your actual API key or ensure it's set in the .env file




# 2- remove the existing directory if it exists
shutil.rmtree('./qdrant_data', ignore_errors=True)  # Remove the existing directory if it exists


# 3- Create a Qdrant client instance
qdrant_client = QdrantClient(path='./qdrant_data')  # Local mode without docker, but only one process can access it at a time


# 4- Create a collection with vector size and distance metric
qdrant_client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE   
    )
)


# Embbeding : 
interpreted_language = ['python', 'javascript', 'ruby', 'php', 'perl', 'lua', 'bash', 'r', 'matlab']
compiled_language = ['c', 'c++', 'java', 'c#', 'go', 'rust', 'swift', 'kotlin', 'typescript']


interpreted_language_response =  [ cohere_client.embed(
    texts=[lang],
    model="embed-english-light-v3.0",
    input_type="classification",
    embedding_types=["float"],
).embeddings.float_[0] for lang in interpreted_language  ]

compiled_language_response =  [ cohere_client.embed(
    texts=[lang],
    model="embed-english-light-v3.0",
    input_type="classification",
    embedding_types=["float"],
).embeddings.float_[0] for lang in compiled_language  ]


"""
Specifies the type of input passed to the model. Required for embedding models v3 and higher.

"search_document": Used for embeddings stored in a vector database for search use-cases.
"search_query": Used for embeddings of search queries run against a vector DB to find relevant documents.
"classification": Used for embeddings passed through a text classifier.
"clustering": Used for the embeddings run through a clustering algorithm.
"image": Used for embeddings with image input.

"""



# 5- Insert points into collections

qdrant_client.upsert(
    collection_name="my_collection",
    wait=True,
    points=[
        PointStruct( 
            id=uuid.uuid4().int,
            vector=interpreted_language_response[i],
            payload={
                "language": interpreted_language[i],
                'type': 'interpreted'}
            ) for i in range(len(interpreted_language_response)
                                 )
]
) 


qdrant_client.upsert(
    collection_name="my_collection",
    wait=True,
    points=[
        PointStruct( 
            id=uuid.uuid4().int,
            vector=compiled_language_response[i],
            payload={
                "language": compiled_language[i],
                'type': 'compiled'}
            ) for i in range(len(compiled_language_response)
                                 )
]
) 


# 6- Query points : 

test_query_embedding = cohere_client.embed(
    texts=["c#"], model="embed-english-light-v3.0", input_type="classification", embedding_types=["float"]).embeddings.float_[0]


results = qdrant_client.query_points(
    collection_name="my_collection",
    query=test_query_embedding,
    limit=3,
    query_filter= Filter(
        must= [FieldCondition( key="type", match=MatchValue(value="interpreted"))] 
        ),
    with_payload=True
    ).points 


# 7- Print the results: 

print("Query Results: ", results) 
