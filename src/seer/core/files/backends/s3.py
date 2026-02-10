"""
S3-compatible file storage backend.

This backend works with AWS S3, Cloudflare R2, MinIO, and other S3-compatible
storage services. Uses boto3 for S3 operations with async wrappers.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from seer.core.files.models import WorkflowFileRef
from seer.core.files.storage import FileNotFoundError as StorageFileNotFoundError, FileStorageBackend, StorageError
from seer.logger import get_logger

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = get_logger("seer.core.files.s3")


class S3FileStorage(FileStorageBackend):
    """
    S3-compatible file storage backend.

    Supports AWS S3, Cloudflare R2, and other S3-compatible services.
    Files are organized as: {prefix}/{user_id}/{run_id}/{file_id}/{filename}

    Example paths:
        - workflow-files/usr_12345/run_abc123/file_xyz789/document.pdf
        - workflow-files/usr_12345/run_abc123/file_xyz789/image.png

    This user-scoped structure provides:
        - Data isolation between users
        - Easy cleanup of all files for a user
        - S3 policy scoping by user prefix if needed
    """

    DEFAULT_PREFIX = "workflow-files"

    def __init__(  # pylint: disable=too-many-arguments  # S3 configuration requires these parameters
        self,
        bucket: str,
        *,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        prefix: str = DEFAULT_PREFIX,
        presigned_url_expiry: int = 3600,
    ):
        """
        Initialize S3 storage backend.

        Args:
            bucket: S3 bucket name.
            region: AWS region (default: us-east-1).
            endpoint_url: Custom endpoint for R2/MinIO (None for AWS S3).
            access_key: AWS access key ID (None to use default credentials).
            secret_key: AWS secret access key (None to use default credentials).
            prefix: Path prefix for all files (default: "workflow-files").
            presigned_url_expiry: Default presigned URL expiration in seconds.
        """
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self.prefix = prefix.strip("/")
        self.presigned_url_expiry = presigned_url_expiry

        # Build client configuration
        client_kwargs: dict[str, Any] = {
            "region_name": region,
            "config": Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }

        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key

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

    def _build_key(self, user_id: str, run_id: str, file_id: str, filename: str) -> str:
        """
        Build the S3 object key.

        Args:
            user_id: User ID for scoping.
            run_id: Workflow run ID.
            file_id: Unique file ID.
            filename: Original filename.

        Returns:
            S3 object key like "workflow-files/usr_123/run_abc/file_xyz/document.pdf"
        """
        # Sanitize filename to prevent path traversal
        safe_filename = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
        # Sanitize user_id and run_id as well (shouldn't contain path separators)
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        safe_run_id = run_id.replace("/", "_").replace("\\", "_")
        return f"{self.prefix}/{safe_user_id}/{safe_run_id}/{file_id}/{safe_filename}"

    def _build_storage_path(self, key: str) -> str:
        """Build the storage_path for a WorkflowFileRef."""
        return f"s3://{self.bucket}/{key}"

    def _parse_storage_path(self, storage_path: str) -> str:
        """Extract the S3 key from a storage_path."""
        # Format: s3://bucket/key
        if not storage_path.startswith("s3://"):
            raise StorageError(f"Invalid storage path format: {storage_path}")
        # Remove s3://bucket/ prefix
        prefix = f"s3://{self.bucket}/"
        if not storage_path.startswith(prefix):
            raise StorageError(f"Storage path bucket mismatch: {storage_path}")
        return storage_path[len(prefix):]

    async def store(
        self,
        user_id: str,
        run_id: str,
        filename: str,
        data: bytes,
        *,
        mime_type: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> WorkflowFileRef:
        """Store file data and return a reference."""
        file_id = str(uuid.uuid4())
        key = self._build_key(user_id, run_id, file_id, filename)

        # Compute MD5 hash for integrity
        md5_hash = hashlib.md5(data).hexdigest()

        # Prepare upload parameters
        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
            "ContentType": mime_type,
        }

        if metadata:
            put_kwargs["Metadata"] = metadata

        try:
            logger.debug("Storing file: bucket=%s key=%s size=%d", self.bucket, key, len(data))
            await self._run_sync(self._client.put_object, **put_kwargs)
        except ClientError as e:
            logger.error("Failed to store file: %s", e)
            raise StorageError(f"Failed to store file: {e}") from e

        return WorkflowFileRef(
            file_id=file_id,
            storage_path=self._build_storage_path(key),
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(data),
            workflow_run_id=run_id,
            created_at=datetime.now(timezone.utc),
            md5_hash=md5_hash,
        )

    async def retrieve(self, file_ref: WorkflowFileRef) -> bytes:
        """Retrieve file data from a reference."""
        key = self._parse_storage_path(file_ref.storage_path)

        try:
            logger.debug("Retrieving file: bucket=%s key=%s", self.bucket, key)
            response = await self._run_sync(
                self._client.get_object, Bucket=self.bucket, Key=key
            )
            # Read the body - this is a StreamingBody
            body = response["Body"]
            data = await self._run_sync(body.read)
            return data
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise StorageFileNotFoundError(f"File not found: {file_ref.file_id}") from e
            logger.error("Failed to retrieve file: %s", e)
            raise StorageError(f"Failed to retrieve file: {e}") from e

    async def retrieve_stream(
        self, file_ref: WorkflowFileRef, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream file data for large files."""
        key = self._parse_storage_path(file_ref.storage_path)

        try:
            logger.debug("Streaming file: bucket=%s key=%s", self.bucket, key)
            response = await self._run_sync(
                self._client.get_object, Bucket=self.bucket, Key=key
            )
            body = response["Body"]

            # Stream in chunks
            while True:
                chunk = await self._run_sync(body.read, chunk_size)
                if not chunk:
                    break
                yield chunk
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise StorageFileNotFoundError(f"File not found: {file_ref.file_id}") from e
            logger.error("Failed to stream file: %s", e)
            raise StorageError(f"Failed to stream file: {e}") from e

    async def delete(self, file_ref: WorkflowFileRef) -> bool:
        """Delete a single file."""
        key = self._parse_storage_path(file_ref.storage_path)

        try:
            # Check if file exists first
            exists = await self.exists(file_ref)
            if not exists:
                return False

            logger.debug("Deleting file: bucket=%s key=%s", self.bucket, key)
            await self._run_sync(self._client.delete_object, Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            logger.error("Failed to delete file: %s", e)
            raise StorageError(f"Failed to delete file: {e}") from e

    async def delete_by_run(self, user_id: str, run_id: str) -> int:
        """Delete all files for a workflow run."""
        prefix = f"{self.prefix}/{user_id}/{run_id}/"
        deleted_count = 0

        try:
            logger.info("Deleting all files for user %s, run %s", user_id, run_id)

            # List all objects with the run prefix
            paginator = self._client.get_paginator("list_objects_v2")

            async def list_and_delete():
                nonlocal deleted_count
                # Run paginator synchronously since it returns an iterator
                for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                    contents = page.get("Contents", [])
                    if not contents:
                        continue

                    # Build delete request
                    objects_to_delete = [{"Key": obj["Key"]} for obj in contents]

                    if objects_to_delete:
                        await self._run_sync(
                            self._client.delete_objects,
                            Bucket=self.bucket,
                            Delete={"Objects": objects_to_delete},
                        )
                        deleted_count += len(objects_to_delete)

            await list_and_delete()
            logger.info("Deleted %d files for user %s, run %s", deleted_count, user_id, run_id)
            return deleted_count
        except ClientError as e:
            logger.error("Failed to delete files for user %s, run %s: %s", user_id, run_id, e)
            raise StorageError(f"Failed to delete files for run: {e}") from e

    async def delete_by_user(self, user_id: str) -> int:
        """Delete all files for a user."""
        prefix = f"{self.prefix}/{user_id}/"
        deleted_count = 0

        try:
            logger.info("Deleting all files for user: %s", user_id)

            # List all objects with the user prefix
            paginator = self._client.get_paginator("list_objects_v2")

            async def list_and_delete():
                nonlocal deleted_count
                for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                    contents = page.get("Contents", [])
                    if not contents:
                        continue

                    # Build delete request
                    objects_to_delete = [{"Key": obj["Key"]} for obj in contents]

                    if objects_to_delete:
                        await self._run_sync(
                            self._client.delete_objects,
                            Bucket=self.bucket,
                            Delete={"Objects": objects_to_delete},
                        )
                        deleted_count += len(objects_to_delete)

            await list_and_delete()
            logger.info("Deleted %d files for user %s", deleted_count, user_id)
            return deleted_count
        except ClientError as e:
            logger.error("Failed to delete files for user %s: %s", user_id, e)
            raise StorageError(f"Failed to delete files for user: {e}") from e

    async def get_presigned_url(
        self,
        file_ref: WorkflowFileRef,
        expires_seconds: int = 3600,
    ) -> str:
        """Get a presigned URL for direct file download."""
        key = self._parse_storage_path(file_ref.storage_path)
        expiry = expires_seconds or self.presigned_url_expiry

        try:
            url = await self._run_sync(
                self._client.generate_presigned_url,
                "get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": key,
                    "ResponseContentDisposition": f'attachment; filename="{file_ref.filename}"',
                },
                ExpiresIn=expiry,
            )
            return url
        except ClientError as e:
            logger.error("Failed to generate presigned URL: %s", e)
            raise StorageError(f"Failed to generate presigned URL: {e}") from e

    async def exists(self, file_ref: WorkflowFileRef) -> bool:
        """Check if a file exists in storage."""
        key = self._parse_storage_path(file_ref.storage_path)

        try:
            await self._run_sync(self._client.head_object, Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            logger.error("Failed to check file existence: %s", e)
            raise StorageError(f"Failed to check file existence: {e}") from e
