from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import re
import pymongo
import base64
import requests
import json
from io import BytesIO

model_type = None
model_pathogen = None
interpreter = None
input_index = None
output_index = None

api_key = "AIzaSyAwEpwCoFSPCAFC7oA3akFLyLejOiknEwQDELETE"
api_Url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

class_names_type = [
    'Apple_Healthy',
    'Apple_notHealthy',
    'Blueberry_Healthy',
    'Cherry_Healthy',
    'Cherry_notHealthy',
    'Corn_Healthy',
    'Corn_notHealthy',
    'Grape_Healthy',
    'Grape_notHealthy',
    'Orange_notHealthy',
    'Peach_Healthy',
    'Peach_notHealthy',
    'Pepper_Healthy',
    'Pepper_notHealthy',
    'Potato_Healthy',
    'Potato_notHealthy',
    'Raspberry_Healthy',
    'Soybean_Healthy',
    'Squash_notHealthy',
    'Strawberry_Healthy',
    'Strawberry_notHealthy',
    'Tomato_Healthy',
    'Tomato_notHealthy'
]
class_names_pathogen = ['Bacteria', 'Fungi', 'Pests', 'Virus']

BUCKET_NAME = "classification-bucktet1" 


def label(image_path):
#   with open(image_path, "rb") as image_file:
#      base64imageData = base64.b64encode(image_file.read()).decode('utf-8')
  base64imageData = base64.b64encode(image_path.getvalue()).decode('utf-8')
     
  requestData = {
      "requests": [
          {
              "image": {
                  "content": base64imageData
              },
              "features": [{"type": "LABEL_DETECTION", "maxResults": 1}]
          }
      ]
  }

  response = requests.post(api_Url, json=requestData)
  data = response.text
  
  response_json = f'''{data}'''

  response_data = json.loads(response_json)

  label = response_data['responses'][0]['labelAnnotations'][0]['description']
  return label

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def dbConnection():
    return pymongo.MongoClient("mongodb+srv://mohamedelaidi41:M36O6w9H00rdfdGL@cluster0.z7kvbwn.mongodb.net/")

def mongo_data_pathogen():
    client = dbConnection()
    plant_db = client["Plant_Info"]
    pathogen_collection = plant_db["pathogen"]

    # Initialize document
    document = []

    # Convert and print documents in JSON format
    for doc in pathogen_collection.find():
        # Convert ObjectId to string for JSON serialization
        doc['_id'] = str(doc['_id'])
        doc.pop('_id', None)
        doc.pop('pathogen', None)
        document.append(doc)
        
    return document

def mongo_data_type(plant):
    client = dbConnection()
    plant_db = client["Plant_Info"]
    nutrition_collection = plant_db["Plant_Nutrition"]

    #plant_regex = re.compile(r'\b{}\b'.format(plant), re.IGNORECASE)

    # Query documents matching the regex pattern
    matching_documents = nutrition_collection.find({"name": plant})

    # Convert and print documents in JSON format
    for document in matching_documents:
        # Convert ObjectId to string for JSON serialization
        document['_id'] = str(document['_id'])
        document.pop('_id', None)
        document.pop('name', None)
        
    return document

def predict(request):
    global model_type
    if model_type is None:
        download_blob(
            BUCKET_NAME,
            "model_simple_cnn_exp2.h5",
            "/tmp/model_simple_cnn_exp2.h5",
        )
        model_type = tf.keras.models.load_model("/tmp/model_simple_cnn_exp2.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((224, 224)) # image resizing
    )

    img_array = tf.expand_dims(image, 0)
    predictions_type = model_type.predict(img_array)
    
    predicted_class_type = class_names_type[np.argmax(predictions_type[0])]
    
    confidence = round(100 * (np.max(predictions_type[0])), 2)
    
    plant = ""
    health_stats = ""
    plant = predicted_class_type.split("_")[0]
    plant_info = mongo_data_type(plant)
    
         # image_arr = np.array(Image.open(BytesIO(image_file)))
        # print(image_arr)
        # image = Image.fromarray(image_arr, 'RGB')
        # image_bytes = BytesIO()
        # image.save(image_bytes, format='JPEG')
        # image = Image.open(image)
        # temp_image_file = tempfile.NamedTemporaryFile(delete=False,suffix='.jpg')
        # temp_image = temp_image_file.name
        # image.save(temp_image)
        # temp_image_file.close()
        # image = Image.open(io.BytesIO(image_file.read()))  
    # labels = label(image_bytes)
    image_arr = Image.fromarray(image, 'RGB')
    image_bytes = BytesIO()
    image_arr.save(image_bytes, format='JPEG')
    labels = label(image_bytes)
    if 'Plant' in  labels or 'Flower' in labels or 'plant' in labels or 'flower' in labels:
        if "not" in predicted_class_type:
            global model_pathogen
            if model_pathogen is None:
                download_blob(
                    BUCKET_NAME,
                    "simple_model_cnn_p.h5",
                    "/tmp/simple_model_cnn_p.h5",
                )
                model_pathogen = tf.keras.models.load_model("/tmp/simple_model_cnn_p.h5")

            health_stats = False
            
            predictions_pathogen = model_pathogen.predict(img_array)
            predicted_class_pathogen = class_names_pathogen[np.argmax(predictions_pathogen[0])]
            
            pathogen_info = mongo_data_pathogen()
            pathogen_info = [item['about'] for item in pathogen_info]

            predicted_class = np.round(predictions_pathogen * 100, 2)
            
            return {
                "confidence_type": str(confidence),
                "status" : True,
                "class": plant,
                "image": plant_info['icon'],
                "health_state": health_stats,
                "Description": plant_info['description'],
                "pathogen": [
                    {
                        'name' : class_names_pathogen[0], 
                        'confidence' : str(predicted_class[0][0]),
                        "about": pathogen_info[0]
                    },
                    {
                        'name' :class_names_pathogen[1],
                        'confidence' : str(predicted_class[0][1]),
                        "about": pathogen_info[1]
                    },
                    {
                        'name' : class_names_pathogen[2], 
                        'confidence' : str(predicted_class[0][2]),
                        "about": pathogen_info[2]
                    },
                    {
                        'name' : class_names_pathogen[3],
                        'confidence' : str(predicted_class[0][3]),
                        "about": pathogen_info[3]
                    }
                ]
            }
        else:
            health_stats = True
            return {
                "confidence_type": str(confidence),
                "status" : True,
                "class": plant,
                "image": plant_info['icon'],
                "health_state": health_stats,
                "Description": plant_info['description'],
                "nutritions": {key: str(value) for key, value in plant_info.items() if(key != 'description' and key != 'icon')}
            }  

    
    else:
        return {
            "result": "No plant detected in the image.",
            "status" : False}