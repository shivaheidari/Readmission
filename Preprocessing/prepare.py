#using data in mongodb
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
list_collection = db.list_collection_names()
admission = db["Admission"]
print(admission.count_documents({}))
noteevents = db["Noteevents"]
#print(noteevents.count_documents({}))
patient_admissions = db["patients_admissions"]
#print(patient_admissions.find_one())
print("begine")
pipeline = [
    # Unwind admissions array
    { "$unwind": { "path": "$admissions", "includeArrayIndex": "index" } },
    
    # Sort admissions by 'admittime'
    { "$sort": { "SUBJECT_ID": 1, "admissions.ADMITIME": 1 } },

    # Group back into arrays but with sorted order
    { "$group": {
        "_id": "$_id",
        "SUBJECT_ID": { "$first": "$SUBJECT_ID" },
        "admissions": { "$push": "$admissions" }
    }},

    # Add a new field to calculate delta_t
    {
        "$addFields": {
            "readmission": {
                "$reduce": {
                    "input": { "$range": [0, { "$size": "$admissions" } ] },
                    "initialValue": False,
                    "in": {
                        "$cond": [
                            { "$lt": [
                                {
                                    "$dateDiff": {
                                        "startDate": { "$arrayElemAt": ["$admissions.dischtime", "$$this"] },
                                        "endDate": { "$arrayElemAt": ["$admissions.admittime", { "$add": ["$$this", 1] }] },
                                        "unit": "day"
                                    }
                                },
                                30
                            ]},
                            True,
                            "$$value"
                        ]
                    }
                }
            }
        }
    }
]

print("end")
results = db.patients_admissions.aggregate(pipeline)

# Save the results to a new collection
db.filtered_patients.insert_many(list(results))




#from patients-admissions collection find patients with readmissions and transfer the data into another collection like patients-readmissions




# from the collection patients-readmissions create a collection like patient; readmission; [notes] * notes comes from NOTEEVENTS refine categories 

# create collection like patiet, concatenated note, label = 1