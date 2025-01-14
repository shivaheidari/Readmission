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



pipeline_concatenate_and_save = [
    # Project the necessary fields and concatenate all note texts
    {
        "$project": {
            "_id": 1,
            "SUBJECT_ID": 1,
            "ADMITTIME": 1,
            "DISCHTIME": 1,
            "readmission": 1,
            "HADM_ID": 1,
            "concatenated_notes": {
                "$reduce": {
                    "input": "$related_notes.text",
                    "initialValue": "",
                    "in": {
                        "$concat": [
                            "$$value",
                            { "$cond": [{ "$eq": ["$$value", ""] }, "", " "] },  # Add a space if not the first note
                            "$$this"
                        ]
                    }
                }
            }
        }
    },
    # Save the results to a new collection
    {
        "$merge": {
            "into": "no_readmitted_concated",
            "whenMatched": "merge",  # Optional: Merge if the document already exists
            "whenNotMatched": "insert"  # Insert if the document doesn't exist
        }
    }
]

# Execute the aggregation pipeline
db.no_readmited_notes.aggregate(pipeline_concatenate_and_save)

print("Data saved to the 'No_readmitted_concated' collection.")
