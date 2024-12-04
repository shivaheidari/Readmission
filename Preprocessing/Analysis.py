#getting know the data  

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
list_collection = db.list_collection_names()
admission = db["Admission"]
print(admission.count_documents({}))
noteevents = db["Noteevents"]
print(noteevents.count_documents({}))
patients_admissions = db["patients_admissions"]



'''
filter data :

hospital readmisiions = unplanned visits within 30 days from earleier state. #admission_TYPE = "EMERGENCY" or "URGENT"Exluce NEWBORNS, death, ...
remove : new newbors, transfers, deaths, and planned readmission: discharge_location: HOME
find nursing notes, radilogy report , physician note, findings reports, lab report, EGG report, and progress not before 24 cut-off
find discharge summary after 24 hours.

in the Noteevents includes discharge summary, category, which includes CATEGORY, 

#subject_ID in ADMISSION and NOTEEVENTS is key
'''




#docs = admission.find({}, {"_id":0, "SUBJECT_ID":1, "ADMISSION_TYPE":1})
#docs = admission.find({"HOSPITAL_EXPIRE_FLAG":1}, {"SUBJECT_ID":1})
# docs = admission.find({"ADMISSION_TYPE": {"$regex":"EMERGENCY"}})
# for doc in docs:
#     print(doc)

# docs = noteevents.find({"SUBJECT_ID": 22})
# for doc in docs:
#     print(doc)

'''
check if how many unique patients there are.
'''
# print(db.list_collection_names())
# # 
# pipeline = [
#     { "$group": { "_id": "$SUBJECT_ID", "admissions":{"$push":"$$ROOT"}}}, {"$project":{"_id":0, "patient_id":"$_id", "admissions":1}}
# ]

# # Execute the aggregation
# results = list(admission.aggregate(pipeline))
# if results:

# # Print the results
#     print("data exists")
#     patients_admissions.insert_many(results)
#     print(patients_admissions.count_documents({}))
#     print(admission.find_one())
# else: 
#     print("nodata")


print(patients_admissions.count_documents({"admissions":{"$size":1}}))

#patients_admissions.delete_many({"admissions":{"$size":1}})
#limit dates

#join the admission and noteevets


