import os, sys
import pymongo
import logging

class MongoDatabase():

    def __init__(self, tag="tmp"):
        constr = "mongodb://127.0.0.1:27017/admin"
        self.client = pymongo.MongoClient(constr, serverSelectionTimeoutMS=60000)
        self.db = self.client.macaoExperimentsDB
        logging.debug("Connected to MongoDB")

    def __del__(self):
        self.client.close()
