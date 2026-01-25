"""
AWS Parameter Store integration for Pydantic Settings.

This module provides a custom Pydantic settings source that reads from AWS SSM Parameter Store
as a fallback when environment variables or .env files don't contain the required values.

Priority order:
1. Environment variables
2. .env file
3. AWS Parameter Store (this module)
4. Default values in Pydantic fields
"""
import os
from typing import Any, Dict, Tuple, Type

import boto3
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from dotenv import load_dotenv
load_dotenv()


class AwsSsmSettingsSource(PydanticBaseSettingsSource):
    """
    Custom Pydantic settings source that reads from AWS SSM Parameter Store.

    This source is designed to be used as a fallback after environment variables
    and .env files. It gracefully handles cases where AWS credentials are not
    available (e.g., local development without AWS setup).

    Parameters are fetched in bulk using get_parameters_by_path for efficiency.
    The parameter path format is: /{environment}/{field_name}
    where environment is read from ENV environment variable (defaults to 'dev').
    """

    def __init__(self, settings_cls: Type[BaseSettings], ssm_path_prefix: str | None = None):
        """
        Initialize the AWS SSM settings source.

        Args:
            settings_cls: The Pydantic BaseSettings class
            ssm_path_prefix: Optional custom path prefix. If None, uses /{ENV}/ format
        """
        super().__init__(settings_cls)
        self._ssm_client = None
        self._cache: Dict[str, Any] | None = None

        # Determine the SSM path prefix
        if ssm_path_prefix is None:
            environment = os.getenv('ENV', 'dev').lower()
            self.ssm_path_prefix = f"/{environment}/"
        else:
            self.ssm_path_prefix = ssm_path_prefix if ssm_path_prefix.endswith('/') else f"{ssm_path_prefix}/"

    def _get_ssm_client(self):
        """Lazily initialize SSM client with error handling."""
        if self._ssm_client is None:
            try:
                # Try to get region from environment, fallback to us-east-1
                region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
                self._ssm_client = boto3.client('ssm', region_name=region)
            except Exception:  # pylint: disable=broad-exception-caught  # Reason: Intentionally catching all exceptions to gracefully handle any AWS setup issues during local development
                # Silently fail if AWS is not configured
                # This allows local development without AWS setup
                pass
        return self._ssm_client

    def _fetch_parameters(self) -> Dict[str, Any]:
        """
        Fetch all parameters from AWS SSM Parameter Store in bulk.

        Returns:
            Dictionary mapping field names to their values from Parameter Store.
            Returns empty dict if AWS is not configured or if fetching fails.
        """
        if self._cache is not None:
            return self._cache

        client = self._get_ssm_client()
        if client is None:
            self._cache = {}
            return self._cache

        parameters = {}
        try:
            # Use paginator to handle large number of parameters
            paginator = client.get_paginator('get_parameters_by_path')
            response_iterator = paginator.paginate(
                Path=self.ssm_path_prefix,
                Recursive=True,
                WithDecryption=True
            )

            for page in response_iterator:
                for param in page.get('Parameters', []):
                    # Extract field name from parameter path
                    # e.g., /dev/openai_api_key -> openai_api_key
                    param_name = param['Name'][len(self.ssm_path_prefix):]
                    # Handle nested paths by replacing / with _
                    # e.g., /dev/database/url -> database_url
                    field_name = param_name.replace('/', '_')
                    parameters[field_name] = param['Value']

            print(f"Parameters fetched from AWS Parameter Store: len(parameters) = {len(parameters)} {parameters.keys()}")
            self._cache = parameters

        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Intentionally catching all exceptions to
            #gracefully handle any AWS parameter store access issues during local development
            # Silently fail and return empty dict
            # This ensures local development works without AWS
            self._cache = {}

        return self._cache

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        """
        Get a single field value from Parameter Store.

        This method is called by Pydantic for each field. However, we override
        __call__ to fetch all parameters at once for efficiency.
        """
        # Not used when __call__ is implemented, but required by abstract base class
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        """
        Return all settings from AWS Parameter Store.

        This is called by Pydantic during settings initialization.
        Returns a dictionary of all available parameters.
        """
        return self._fetch_parameters()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(path_prefix={self.ssm_path_prefix!r})"
