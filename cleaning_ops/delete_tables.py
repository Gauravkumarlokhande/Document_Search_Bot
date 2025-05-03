
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData
import os

load_dotenv()

DATABASE_URL ="postgresql://postgres:TuAststpGHielqgNeHtAgTVgYhBAJLfl@caboose.proxy.rlwy.net:29062/railway"

def drop_all_tables():
    try:
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL)

        # Reflect the database schema
        metadata = MetaData()
        metadata.reflect(bind=engine)

        # Drop all tables
        print("Dropping all tables...")
        metadata.drop_all(bind=engine)
        print("All tables dropped successfully.")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    drop_all_tables()
