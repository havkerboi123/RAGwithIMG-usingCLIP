import os
import numpy as np
from PIL import Image
import base64
import chromadb

from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chromadb.utils.data_loaders import ImageLoader

# Images
img1 = "/Users/mhmh/Desktop/clip/1.png"
img2 = "/Users/mhmh/Desktop/clip/2.png"

chroma_client = chromadb.PersistentClient(path="/Users/mhmh/Desktop/clip/gdsc")
# Instantiate the ChromaDB Image Loader
image_loader = ImageLoader()
# Instantiate CLIP embeddings
CLIP = OpenCLIPEmbeddingFunction()

# Create the image vector database
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function = CLIP, data_loader = image_loader)


image_vdb.add(
    ids=["1","2"],
    uris=[img1,img2]
)

result=image_vdb.query(query_texts="what color was the guys shirt",n_results=1,include=['uris'])

imgg=result['uris'][0]
print(imgg)

import base64

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Specify the image path correctly
imgg = "/Users/mhmh/Desktop/clip/1.png"  # Example path

# Encode image to base64
base64_image = encode_image(imgg)

from openai import OpenAI

client = OpenAI(api_key="")
MODEL = "gpt-4o" 
user_query="what color was the guys shirt"



response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an helpful Ai assistant that answers user queries with the help of relevant image provided."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Answer the user query with the help of image provided{user_query}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]}
                ],
                temperature=0.0,
            )


response.choices[0].message.content.strip()



        


