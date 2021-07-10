from typing import Optional

import os

import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings

from pymongo import MongoClient, database, collection, cursor, settings
from bson.objectid import ObjectId
import base64

import numpy as np

import cv2
import gridfs
import pytz


class Settings(BaseSettings):
    mongo_host: str = '127.0.0.1'
    mongo_port: int = 27017
    
    class Config:
        env_file = ".env"


class TimeFilter(BaseModel):
    lower: float
    upper: float

class Item(BaseModel):
    tracking_id: int
    first_timestamp: float
    framepath: str

settings = Settings()
mongo_client = None
app = FastAPI()
db = None
fs = None

origins = [
    "http://localhost",
    "http://127.0.0.1"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():

    global db
    global fs

    mongo_client =  MongoClient(settings.mongo_host,settings.mongo_port,connect=True)

    db = mongo_client.get_database('people_project')

    fs = gridfs.GridFS(db)

    print(db.list_collection_names())

   #print(db['people'].insert_one({'person':12}))




@app.get("/")
async def root():

    records = db['person'].find(limit=0)
    return_dict = []
    for item in records:
         # get the image from gridfs
        gOut = fs.get(ObjectId(item['image_id']))

        # get the image from gridfs
        gOut = fs.get(ObjectId(item['image_id']))

        # convert bytes to ndarray
        img = np.frombuffer(gOut.read(), dtype=np.uint8)

        shape = item['image_shape']

        img = np.reshape(img, np.array([ shape[0], shape[1], 3 ]))

        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)

        time = datetime.datetime.fromtimestamp(int(item['first_timestamp'])).strftime(format="%Y-%m-%d %H:%M:%S")
        return_dict.append({'tracking_id':item['tracking_id'],'timestamp':time,'image':jpg_as_text})
    return return_dict#{"message": "Hello World"}

@app.post("/filtertime/")
async def filtertime(timefilter: TimeFilter):
    print(timefilter.lower/1000)
    print(timefilter.upper/1000)
    records = db['person'].find({"first_timestamp":{"$gte":timefilter.lower/1000, "$lt":timefilter.upper/1000}},limit=0)
    print(records.count())
    #print(records)
    return_dict = []
    for item in records:
        # get the image from gridfs
        gOut = fs.get(ObjectId(item['image_id']))

        # convert bytes to ndarray
        img = np.frombuffer(gOut.read(), dtype=np.uint8)

        shape = item['image_shape']

        img = np.reshape(img, np.array([ shape[0], shape[1], 3 ]))

        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)

        #jpg_as_text = base64.b64encode(img)
        
        # Localize the timezone to Stockholm'
        tz = pytz.timezone('Europe/Stockholm')
        time = pytz.utc.localize(datetime.datetime.fromtimestamp(int(item['first_timestamp']))).astimezone(tz).strftime(format="%Y-%m-%d %H:%M:%S")
        return_dict.append({'tracking_id':item['tracking_id'],'timestamp':time,'image':jpg_as_text})
    return return_dict#{"message": "Hello World"}


@app.post("/items/")
async def create_item(item: Item):
    return item
