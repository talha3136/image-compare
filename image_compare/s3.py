from django.conf import settings
import boto3
# from botocore.config import Config


class S3:
    def __init__(self):
        self.client = boto3.client('s3',
                                   settings.AWS_REGION,
                                   aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                   aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                   )

    def delete_file(self, key):
        print("before file delete")
        return self.client.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=key)
