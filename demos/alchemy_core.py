from sqlalchemy import create_engine, MetaData, Table, Integer,String,Column
from dotenv import load_dotenv
import os


load_dotenv()
db_url=os.getenv("DB_URL")


engine = create_engine(db_url,echo=True)

''' Note : the previous code was a basic usage of sql alchemy
but we use sqlalchemy to work with sqlalchemy core where we can use functions 
and metadata to create everything in database easily
or we can use sqlalchemy orm (object relational  mapper) to map 
python classes or objects to database '''

'''metadata keeps track of the information needed for creation of table'''

meta=MetaData()
people=Table("people2",meta,Column("id", Integer, primary_key=True),
Column("name",String,nullable=False))
meta.create_all(engine)

''' one more interesting thing is that we switch to any database by just switching the url'''

conn=engine.connect()
# we can use functions to interact with the table
insert_st=people.insert().values(name='jabba',id=1)
result=conn.execute(insert_st)
conn.commit()

# similarly we can perform other sql operations
# for row in result.fetchall():
#     print(row)
# for this code we need a select statement similar to insert statement

