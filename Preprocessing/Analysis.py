#getting know the data  

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
list_collection = db.list_collection_names()
admission = db["Admission"]
print(admission.count_documents({}))
noteevents = db["Noteevents"]
print(noteevents.count_documents({}))


#find the number of patients
'''

hospital readmisiions = unplanned visits within 30 days from earleier state. #admission_TYPE = "EMERGENCY" or "URGENT"Exluce NEWBORNS, death, ...
'''

'''

remove : new newbors, transfers, deaths, and planned readmission: discharge_location: HOME
'''


