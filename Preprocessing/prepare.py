#using data in mongodb
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
list_collection = db.list_collection_names()

collection = db["Admission"]
print(collection.count_documents({}))
collection = db["Noteevents"]
print(collection.count_documents({}))