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
print(db.list_collection_names())
readmission_status = db["readmission_status"]
readmitted = db["readmitted"]
print("readmited_counts",readmitted.count_documents({}))
noreadmited = db["no_readmited"]
print("noradmited_counts", noreadmited.count_documents({}))


#pipelines
#-----------------------------------------------------------------------

pipeline_true_readmission = [
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


pipeline_readmitted = [{ 
        "$addFields": {
            "DISCHTIME": { "$toDate": "$DISCHTIME" },
            "ADMITTIME": { "$toDate": "$ADMITTIME" }
        }
    },
    # Self-join the collection to find other admissions for the same SUBJECT_ID
    {   
        "$lookup": {
            "from": "Admission",
            "localField": "SUBJECT_ID",
            "foreignField": "SUBJECT_ID",
            "as": "related_admissions"
        }
    },
    # Unwind related admissions to compare each admission pair
    {
        "$unwind": "$related_admissions"
    },
    # Filter to include only pairs where the new admission occurs after the current discharge
    {
        "$match": {
            "$expr": {
                "$and": [
                    { "$ne": ["$ADMITTIME", "$related_admissions.ADMITTIME"] },
                    { "$lt": ["$DISCHTIME", "$related_admissions.DISCHTIME"] }
                ]
            }
        }
    },
    # Convert string date fields to Date type
   
    # Calculate the difference in days between discharge and the next admission
    {
        "$addFields": {
            "delta_t": {
                "$dateDiff": {
                    "startDate": "$DISCHTIME",
                    "endDate": "$related_admissions.ADMITTIME",
                    "unit": "day"
                }
            }
        }
    },
    # Group by the original document to determine if any readmission occurs within 30 days
    {
        "$group": {
            "_id": "$_id",
            "SUBJECT_ID": { "$first": "$SUBJECT_ID" },
            "ADMITTIME": { "$first": "$ADMITTIME" },
            "DISCHTIME": { "$first": "$DISCHTIME" },
            "readmission": {
                "$push": {
                    "$cond": [{ "$lte": ["$delta_t", 30] }, True, False]
                }
            }
        }
    },
    # Add a final field to summarize if any readmission is within 30 days
    {
        "$addFields": {
            "readmission": { "$anyElementTrue": "$readmission" }
        }
    },
    # Remove unnecessary intermediate fields
    {
        "$project": {
            "_id": 1,
            "SUBJECT_ID": 1,
            "ADMITTIME": 1,
            "DISCHTIME": 1,
            "readmission": 1,
            
        }
    }
]

pipeline_readmission_notes = [
{"$lookup":{
    "from":"Noteevents",
    "localField":"HADM_ID", 
    "foreignField": "HADM_ID",
    "as": "related_notes"
}},
{
        "$project": {
            "_id": 1,
            "SUBJECT_ID": 1,
            "ADMITTIME": 1,
            "DISCHTIME": 1,
            "readmission": 1,
            "HADM_ID": 1,
             "related_notes": {
                "$map": {
                    "input": "$related_notes",    # Iterate over each note in related_notes
                    "as": "note",                 # Alias for each element
                    "in": {                       # Define the shape of each note
                        "text": "$$note.TEXT",    # Include TEXT from each note
                        "note_type": "$$note.CATEGORY"  # Include CATEGORY from each note
                    }
                }
            }

            
        }
    }

]



pipeline_no_readmission_notes = [
{"$lookup":{
    "from":"Noteevents",
    "localField":"HADM_ID", 
    "foreignField": "HADM_ID",
    "as": "related_notes"
}},
{
        "$project": {
            "_id": 1,
            "SUBJECT_ID": 1,
            "ADMITTIME": 1,
            "DISCHTIME": 1,
            "readmission": 1,
            "HADM_ID": 1,
             "related_notes": {
                "$map": {
                    "input": "$related_notes",    # Iterate over each note in related_notes
                    "as": "note",                 # Alias for each element
                    "in": {                       # Define the shape of each note
                        "text": "$$note.TEXT",    # Include TEXT from each note
                        "note_type": "$$note.CATEGORY"  # Include CATEGORY from each note
                    }
                }
            }

            
        }
    }

]


results = noreadmited.aggregate(pipeline_no_readmission_notes)
db.no_readmited_notes.insert_many(results)
print("inserted...")
#-------------------------End of pipelines----------------------------------------------------

#--------------aggregations------------------
