# object relational mapper
# we can map python classes and objects to sql 
# here we need a session and not a connection
from sqlalchemy import create_engine, Integer, String, Float, Column, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv
import os

load_dotenv()
# relation for two different classes
# tables as classes and rows as objects
db_url=os.getenv("DB_URL")
engine=create_engine(db_url,echo=True)
# echo is to see the logs going behind

base = declarative_base()

class Person(base): 
# this is how you define table class
    __tablename__="people3" # this is table name
    id =Column(Integer,primary_key=True)
    name=Column(String,nullable=False)
    age=Column(Integer)

    things=relationship('Thing',back_populates='person')

class Thing(base):
    __tablename__='things'
    id =Column(Integer,primary_key=True)
    description=Column(String,nullable=True)
    value=Column(Float)
    owner=Column(Integer,ForeignKey('people3.id'))

    person=relationship('Person',back_populates='things')

'''this relationship creates relationship between these two tables.
and the backpopulate is for a two way connection'''

base.metadata.create_all(engine) # to create table
Session=sessionmaker(bind=engine)
session=Session()
new_person=Person(name='same',age=80,id=1)
session.add(new_person)
session.commit()

