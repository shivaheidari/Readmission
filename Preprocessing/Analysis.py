#getting know the data  

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
list_collection = db.list_collection_names()
admission = db["Admission"]
print(admission.count_documents({}))
noteevents = db["Noteevents"]
print(noteevents.count_documents({}))


'''
filter data :

hospital readmisiions = unplanned visits within 30 days from earleier state. #admission_TYPE = "EMERGENCY" or "URGENT"Exluce NEWBORNS, death, ...
remove : new newbors, transfers, deaths, and planned readmission: discharge_location: HOME
find nursing notes, radilogy report , physician note, findings reports, lab report, EGG report, and progress not before 24 cut-off
find discharge summary after 24 hours.

in the Noteevents includes discharge summary, category, which includes CATEGORY, 

#subject_ID in ADMISSION and NOTEEVENTS is key
'''



#find query

docs = admission.find({"SUBJECT_ID" : 22})
for doc in docs:
    print(doc)
docs = noteevents.find({"SUBJECT_ID": 22})
for doc in docs:
    print(doc)