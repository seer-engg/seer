import os
from seer.logger import get_logger
logger = get_logger(__name__)


def _ensure_mlflow_autologging() -> None:
    """Enable MLflow LangChain autologging once per process with tracking URI and experiment."""

    try:
        import mlflow  # pylint: disable=import-outside-toplevel  # Reason: lazy loading to avoid dependency if not used
        from mlflow.langchain import autolog  # pylint: disable=import-outside-toplevel  # Reason: lazy loading to avoid dependency if not used
    except ImportError:
        logger.warning("mlflow not installed, skipping mlflow autologging for langchain")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "seer-workflow-agent")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        autolog()
        logger.info("Enabled MLflow LangChain  autologging (tracking_uri=%s, experiment=%s)", tracking_uri, experiment_name)
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: instrumentation should not break agent startup
        logger.warning("Failed to enable MLflow autologging: %s", exc)
