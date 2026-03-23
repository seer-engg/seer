"""
Oura Ring health data tools — heart rate, sleep, readiness, activity, SpO2, stress, workouts, temperature.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.oura.base import OURA_API_BASE, OuraAPIClient

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.oura.health")


def _build_date_params(arguments: Dict[str, Any]) -> Dict[str, str]:
    """Build Oura API date query params from tool arguments."""
    today = date.today().isoformat()
    params: Dict[str, str] = {}
    params["start_date"] = arguments.get("start_date") or today
    params["end_date"] = arguments.get("end_date") or today
    return params


class OuraGetHeartRateTool(OuraAPIClient):
    """Get heart rate readings from Oura Ring."""

    name = "oura_get_heart_rate"
    description = "Get heart rate readings from Oura Ring for a date range. Returns individual HR samples with timestamps."
    required_scopes = ["heartrate"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of heart rate samples"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/heartrate", credentials=credentials, params=params)
        return resp.json()


class OuraGetSleepTool(OuraAPIClient):
    """Get detailed sleep session data from Oura Ring."""

    name = "oura_get_sleep"
    description = "Get detailed sleep data from Oura Ring — stages, efficiency, HR, HRV, and sleep scores."
    required_scopes = ["daily"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of sleep sessions with stages, scores, and metrics"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/sleep", credentials=credentials, params=params)
        return resp.json()


class OuraGetReadinessTool(OuraAPIClient):
    """Get daily readiness scores from Oura Ring."""

    name = "oura_get_readiness"
    description = "Get daily readiness/recovery scores from Oura Ring with contributor breakdown."
    required_scopes = ["daily"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of daily readiness scores"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/daily_readiness", credentials=credentials, params=params)
        return resp.json()


class OuraGetActivityTool(OuraAPIClient):
    """Get daily activity data from Oura Ring."""

    name = "oura_get_activity"
    description = "Get daily activity data from Oura Ring — steps, calories, and intensity breakdown."
    required_scopes = ["daily"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of daily activity summaries"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/daily_activity", credentials=credentials, params=params)
        return resp.json()


class OuraGetSpO2Tool(OuraAPIClient):
    """Get blood oxygen (SpO2) data from Oura Ring."""

    name = "oura_get_spo2"
    description = "Get blood oxygen saturation (SpO2) readings from Oura Ring."
    required_scopes = ["spo2"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of SpO2 readings"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/daily_spo2", credentials=credentials, params=params)
        return resp.json()


class OuraGetStressTool(OuraAPIClient):
    """Get stress level data from Oura Ring."""

    name = "oura_get_stress"
    description = "Get daily stress levels and recovery time from Oura Ring."
    required_scopes = ["stress"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of daily stress readings"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/daily_stress", credentials=credentials, params=params)
        return resp.json()


class OuraGetWorkoutsTool(OuraAPIClient):
    """Get workout sessions from Oura Ring."""

    name = "oura_get_workouts"
    description = "Get workout sessions from Oura Ring with type, duration, and intensity."
    required_scopes = ["workout"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of workout sessions"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/workout", credentials=credentials, params=params)
        return resp.json()


class OuraGetTemperatureTool(OuraAPIClient):
    """Get body temperature data from Oura Ring."""

    name = "oura_get_temperature"
    description = "Get daily body temperature deviation from Oura Ring. Useful for illness and cycle detection."
    required_scopes = ["daily"]
    integration_type = "oura"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.date_range_params_schema(),
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Array of daily temperature readings"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        params = _build_date_params(arguments)
        resp = await self._make_request("GET", f"{OURA_API_BASE}/daily_temperature", credentials=credentials, params=params)
        return resp.json()
