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

# List objects information.
objects = client.list_objects("dsva")
for obj in objects:
    print(obj)

for root,sub_folder, files in os.walk(r"/root/DSVA/processed_data"):
    for filename in files:
        folder_prefix = "data/"
        client.fput_object("dsva",os.path.join(folder_prefix,filename),os.path.join(root,filename))

