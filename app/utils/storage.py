import os
import logging
from fastapi.concurrency import run_in_threadpool
from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageClient:
    def __init__(self):
        self.bucket = settings.S3_BUCKET
        self.endpoint = settings.S3_ENDPOINT
        self.access_key = settings.S3_ACCESS_KEY
        self.secret_key = settings.S3_SECRET_KEY
        self.use_s3 = all([self.bucket, self.access_key, self.secret_key])
        
        self.local_dir = os.path.join(os.getcwd(), "storage", "images")

    async def upload_image(self, filename: str, content: bytes) -> str:
        """
        Uploads image content. Returns the storage path/URL.
        """
        if self.use_s3:
            try:
                return await self._upload_to_s3(filename, content)
            except Exception as e:
                logger.error(f"S3 upload failed: {str(e)}. Falling back to local storage.")

        return await self._upload_to_local(filename, content)

    async def _upload_to_s3(self, filename: str, content: bytes) -> str:
        import boto3
        from botocore.client import Config

        session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        
        endpoint_url = self.endpoint if self.endpoint else None
        
        s3 = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        
        def _upload():
            s3.put_object(Bucket=self.bucket, Key=filename, Body=content)
            if self.endpoint:
                return f"{self.endpoint}/{self.bucket}/{filename}"
            return f"https://{self.bucket}.s3.amazonaws.com/{filename}"
            
        return await run_in_threadpool(_upload)

    async def _upload_to_local(self, filename: str, content: bytes) -> str:
        filepath = os.path.join(self.local_dir, filename)
        
        def _save():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(content)
            return filepath
            
        await run_in_threadpool(_save)
        return filepath.replace("\\", "/")
