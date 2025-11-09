# template.py
from e2b import Template, default_build_logger
from sandbox.constants import BASE_TEMPLATE_ALIAS, BASE_TEMPLATE_CPU_COUNT, BASE_TEMPLATE_MEMORY_MB
from shared.logger import get_logger

def ensure_base_template():
    BASE_TEMPLATE = (
        Template()
        .from_base_image()
        .run_cmd("sudo apt install lsof")
        .run_cmd("sudo apt install net-tools")
    )
    Template.build(
        BASE_TEMPLATE,
        alias=BASE_TEMPLATE_ALIAS,
        cpu_count=BASE_TEMPLATE_CPU_COUNT,
        memory_mb=BASE_TEMPLATE_MEMORY_MB,
        on_build_logs=default_build_logger(),
    )
    return True
