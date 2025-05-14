from minio import Minio
import os
from dotenv import load_dotenv
load_dotenv("config_loader/.env")

access=os.getenv("MINIO_ACCESS_KEY")
secret=os.getenv("MINIO_SECRET_KEY")

client = Minio(
    "192.168.1.5:9000",
    access_key=access,
    secret_key=secret,
    secure=False
    
)
# client.make_bucket("my-bucket")
buckets = client.list_buckets()
for bucket in buckets:
    print(bucket.name, bucket.creation_date)