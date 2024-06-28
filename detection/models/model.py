from peewee import *

db = SqliteDatabase('vehicles.sqlite')

class VehicleModel(Model):
    number_plate = CharField()
    emission_done = BooleanField()
    
    def __str__(self): 
        return self.number_plate
    
    class Meta:
        database = db # This model uses the "people.db" database.

db.create_tables([VehicleModel])
try:
    db.connect()
    print("Connected to database")
except Exception as e:
    print(e)
