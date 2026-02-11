"""
HITL Form Service - Creates temporary forms for Human-in-the-Loop workflow responses.

This service generates form-based HITL collection by creating TriggerSubscription
records that the form router can serve. When a user submits the form, the form
router detects the HITL metadata and auto-resumes the workflow.
"""

from typing import Any, Dict, List, Tuple

from seer.config import config
from seer.database import TriggerSubscription, WorkflowRun
from seer.logger import get_logger

logger = get_logger(__name__)


class HITLFormService:
    """
    Service for creating temporary forms that collect HITL responses.

    Instead of parsing email replies or requiring users to use the platform API,
    this service creates hosted forms that users can fill out. The form submission
    automatically resumes the interrupted workflow.
    """

    async def create_hitl_form(
        self,
        workflow_run: WorkflowRun,
        interrupt_data: Dict[str, Any],
    ) -> Tuple[TriggerSubscription, str]:
        """
        Create a temporary form for HITL response collection.

        Args:
            workflow_run: The interrupted workflow run
            interrupt_data: The HITL interrupt payload from LangGraph

        Returns:
            Tuple of (TriggerSubscription, form_url)
        """
        node_id = interrupt_data.get("node_id", "unknown")

        # Generate unique suffix: hitl-{run_id}-{node_id}
        # Using run_id (public ID like run_123) ensures uniqueness
        suffix = f"hitl-{workflow_run.run_id}-{node_id}"

        # Map HITL inputs to form fields
        hitl_inputs = interrupt_data.get("inputs", [])
        form_fields = self._map_hitl_inputs_to_form_fields(hitl_inputs)

        # Build form config with HITL metadata for callback
        form_config = {
            "title": interrupt_data.get("title", "Action Required"),
            "description": interrupt_data.get("description"),
            "submitButtonText": "Submit Response",
            "successMessage": "Your response has been recorded. The workflow will continue.",
            # HITL metadata - used by form submission handler to auto-resume
            "_hitl_run_id": workflow_run.run_id,
            "_hitl_node_id": node_id,
            "_hitl_display": interrupt_data.get("display", []),
        }

        # Ensure workflow is loaded for the foreign key
        await workflow_run.fetch_related("workflow", "user")

        # Create TriggerSubscription (temporary, for HITL only)
        subscription = await TriggerSubscription.create(
            trigger_key="form.hitl",
            trigger_id=f"hitl-form-{workflow_run.run_id}-{node_id}",
            title=f"HITL Form: {interrupt_data.get('title', 'Response')}",
            user=workflow_run.user,
            workflow=workflow_run.workflow,
            enabled=True,
            form_suffix=suffix,
            form_fields=form_fields,
            form_config=form_config,
        )

        form_url = f"{config.frontend_url}/forms/{suffix}"

        logger.info(
            "Created HITL form for run '%s' node '%s': %s",
            workflow_run.run_id,
            node_id,
            form_url,
            extra={
                "run_id": workflow_run.run_id,
                "node_id": node_id,
                "form_suffix": suffix,
                "subscription_id": subscription.id,
            },
        )

        return subscription, form_url

    def _map_hitl_inputs_to_form_fields(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map HITL input field definitions to form field format.

        Args:
            inputs: List of HITLInputField dicts from the interrupt payload

        Returns:
            List of form field definitions compatible with the form router
        """
        form_fields = []
        for inp in inputs:
            field: Dict[str, Any] = {
                "name": inp.get("id", ""),
                "displayLabel": inp.get("question", ""),
                "type": self._map_input_type(inp.get("input_type", "text")),
                "required": inp.get("required", True),
            }

            # Add placeholder if provided
            placeholder = inp.get("placeholder")
            if placeholder:
                field["placeholder"] = placeholder

            # Add default value if provided
            default_value = inp.get("default_value")
            if default_value is not None:
                field["defaultValue"] = default_value

            # Handle choice options for select/multiselect types
            options = inp.get("options")
            if options:
                field["options"] = [
                    {"value": opt.get("value", ""), "label": opt.get("label", "")}
                    for opt in options
                ]

            form_fields.append(field)

        return form_fields

    def _map_input_type(self, hitl_type: str) -> str:
        """
        Map HITL input types to form field types.

        Args:
            hitl_type: The HITLInputType enum value

        Returns:
            Form field type string
        """
        mapping = {
            "text": "text",
            "number": "number",
            "boolean": "checkbox",
            "single_choice": "select",
            "multi_choice": "multiselect",
        }
        return mapping.get(hitl_type, "text")

    async def disable_hitl_form(self, subscription_id: int) -> None:
        """
        Disable an HITL form after it has been used.

        HITL forms are one-time use - once submitted, they should be disabled
        to prevent duplicate workflow resumes.

        Args:
            subscription_id: The TriggerSubscription ID to disable
        """
        await TriggerSubscription.filter(id=subscription_id).update(enabled=False)
        logger.debug("Disabled HITL form subscription %d", subscription_id)
