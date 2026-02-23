"""
S3 client for seed data operations.

Follows patterns from src/seer/core/files/backends/s3.py:
- Uses boto3 with async wrappers via run_in_executor
- Reuses workflow_file_s3_bucket configuration
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from seer.config import config
from seer.logger import get_logger

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = get_logger("seer.database.seed.s3")

SEED_DATA_PREFIX = "seed-data"
DEFAULT_SEED_KEY = "oauth-seed-data.json"


class SeedS3Client:
    """S3 client for seed data upload/download operations."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        """
        Initialize S3 client for seed data.

        Uses boto3's default credential chain for AWS credentials.

        Args:
            bucket: S3 bucket name. Defaults to workflow_file_s3_bucket config.
            endpoint_url: Custom endpoint for R2/MinIO. Defaults to workflow_file_s3_endpoint_url config.

        Raises:
            ValueError: If bucket is not configured.
        """
        self.bucket = bucket or config.workflow_file_s3_bucket
        self.endpoint_url = endpoint_url or config.workflow_file_s3_endpoint_url

        if not self.bucket:
            raise ValueError(
                "S3 bucket not configured. Set WORKFLOW_FILE_S3_BUCKET environment variable."
            )

        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

        client_kwargs: dict[str, Any] = {
            "region_name": region,
            "config": Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }

        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        self._client: S3Client = boto3.client("s3", **client_kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_event_loop()
        return self._loop

    async def _run_sync(self, func, *args, **kwargs) -> Any:
        """Run a synchronous boto3 call in the thread pool."""
        loop = self._get_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    def _build_key(self, filename: str = DEFAULT_SEED_KEY) -> str:
        """Build S3 key for seed data."""
        return f"{SEED_DATA_PREFIX}/{filename}"

    async def upload_seed_data(self, data: dict, filename: str = DEFAULT_SEED_KEY) -> str:
        """
        Upload seed data JSON to S3.

        Args:
            data: Dictionary to serialize as JSON.
            filename: Filename within seed-data/ prefix.

        Returns:
            S3 URI (s3://bucket/key).

        Raises:
            ClientError: If upload fails.
        """
        key = self._build_key(filename)
        body = json.dumps(data, indent=2, default=str).encode("utf-8")

        try:
            logger.info("Uploading seed data to s3://%s/%s (%d bytes)", self.bucket, key, len(body))
            await self._run_sync(
                self._client.put_object,
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        except ClientError as e:
            logger.error("Failed to upload seed data: %s", e)
            raise

        s3_uri = f"s3://{self.bucket}/{key}"
        logger.info("Seed data uploaded: %s", s3_uri)
        return s3_uri

    async def download_seed_data(self, filename: str = DEFAULT_SEED_KEY) -> dict:
        """
        Download seed data JSON from S3.

        Args:
            filename: Filename within seed-data/ prefix.

        Returns:
            Parsed JSON dictionary.

        Raises:
            FileNotFoundError: If seed data doesn't exist.
            ClientError: If download fails for other reasons.
        """
        key = self._build_key(filename)

        try:
            logger.info("Downloading seed data from s3://%s/%s", self.bucket, key)
            response = await self._run_sync(
                self._client.get_object,
                Bucket=self.bucket,
                Key=key,
            )
            # Read body synchronously (it's already fetched)
            body = response["Body"].read()
            return json.loads(body.decode("utf-8"))
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Seed data not found: s3://{self.bucket}/{key}") from e
            logger.error("Failed to download seed data: %s", e)
            raise

    async def seed_data_exists(self, filename: str = DEFAULT_SEED_KEY) -> bool:
        """
        Check if seed data exists in S3.

        Args:
            filename: Filename within seed-data/ prefix.

        Returns:
            True if file exists, False otherwise.
        """
        key = self._build_key(filename)

        try:
            await self._run_sync(
                self._client.head_object,
                Bucket=self.bucket,
                Key=key,
            )
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise
