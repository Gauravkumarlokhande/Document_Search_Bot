import pandas as pd
from slqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
db_url=os.getenv("DB_URL")
engine=create_engine(db_url,echo=True)
df=pd.read_sql("SELECT * FROM people",con=engine)
print(df)