#using data in mongodb
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["MIMIC"]
list_collection = db.list_collection_names()
admission = db["Admission"]
print(admission.count_documents({}))
noteevents = db["Noteevents"]
print(noteevents.count_documents({}))
patient_admissions = db["patients_admissions"]
#print(patient_admissions.find_one())

#

# pipeline = [
    
#     #lookup previous admission for each patient 
    
#     {"$lookup":{"from":"admission", 
#                 "localField":"SUBJECT_ID",
#                 "foreignField":"SUBJECT_ID",
#                 "as":"previous_admissions"}}]
# results = admission.aggregate(pipeline)

# for doc in results:
#     print(doc)


pipeline = [
    # Lookup previous admission for each patient
    {
        "$lookup": {
            "from": "Admission",  # Join with the same 'admission' collection
            "localField": "SUBJECT_ID",  # Field in current collection
            "foreignField": "SUBJECT_ID",  # Field in 'admission' collection
            "as": "previous_admissions"  # Output field name for the array
        }
    },

    {
        "$unwind": {
            "path": "$previous_admissions",
            "preserveNullAndEmptyArrays": True  # This ensures you don't lose documents with no match
        }
    },


    {
        "$addFields": {
            "time_difference": {
                "$divide": [
                    { 
                        "$subtract": [
                            { "$toDate": "$admittime" },  # Current admission time
                            { "$toDate": "$previous_admissions.dischtime" }  # Previous discharge time
                        ]
                    },
                    1000 * 60 * 60 * 24  # Convert milliseconds to days
                ]
            }
        }
    },
 
    {
        "$match": {
            "time_difference": { "$gt": 0, "$lte": 30 } 
        }
    }
]

# Execute the aggregation
results = admission.aggregate(pipeline)

# Check the results
for doc in results:
    print(doc)




#from patients-admissions collection find patients with readmissions and transfer the data into another collection like patients-readmissions




# from the collection patients-readmissions create a collection like patient; readmission; [notes] * notes comes from NOTEEVENTS refine categories 

# create collection like patiet, concatenated note, label = 1