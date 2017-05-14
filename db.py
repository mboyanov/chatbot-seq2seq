from pymongo import MongoClient
client = MongoClient()
db = client.get_database("matched_pairs")