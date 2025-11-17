"""Variable injection utilities for test runner."""
import re
from typing import Any, Dict


def get_field(obj: Any, field_path: str) -> Any:
    """Get nested field using dot notation and array indexing."""
    if not obj or not field_path:
        return obj
    try:
        parts = re.findall(r'(\w+)|\[(\d+)\]', field_path)
        keys = [match[0] or match[1] for match in parts]
        
        value = obj
        for key in keys:
            if not key:
                continue
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)] if int(key) < len(value) else None
            else:
                return None
        return value
    except Exception:
        return None


def inject_variables(params_data: Any, variables: Dict[str, Any], mcp_resources: Dict[str, Any]) -> Any:
    """Inject [var:...] and [resource:...] tokens into parameters."""
    if isinstance(params_data, dict):
        return {
            k: inject_variables(v, variables, mcp_resources) 
            for k, v in params_data.items()
        }
    if isinstance(params_data, list):
        return [
            inject_variables(item, variables, mcp_resources) 
            for item in params_data
        ]
    if isinstance(params_data, str):
        # Fail fast on invalid placeholders
        if re.match(r'^<[^>]+>$', params_data):
            raise ValueError(f"Invalid placeholder '{params_data}'. Use [var:name] or [resource:name]")
        
        # Full-string [var:...] replacement
        var_match = re.match(r"^\[var:([\w\.\[\]]+)\]$", params_data)
        if var_match:
            var_path = var_match.group(1)
            if '.' in var_path or '[' in var_path:
                base_var = re.match(r'^(\w+)', var_path).group(1)
                return get_field(variables.get(base_var), var_path[len(base_var):].lstrip('.'))
            return variables.get(var_path)
        
        # Full-string [resource:...] replacement
        res_match = re.match(r"^\[resource:(\w+)(?:\.(.+))?\]$", params_data)
        if res_match:
            res_key, field_path = res_match.groups()
            resource = mcp_resources.get(res_key, {})
            return get_field(resource, field_path) if field_path else resource.get('id')

        # Inline replacement
        def replace_var(m):
            var_path = m.group(1)
            if '.' in var_path or '[' in var_path:
                base_var = re.match(r'^(\w+)', var_path).group(1)
                result = get_field(variables.get(base_var), var_path[len(base_var):].lstrip('.'))
                return str(result) if result is not None else ''
            return str(variables.get(var_path, ''))
        
        injected = re.sub(r'\[var:([\w\.\[\]]+)\]', replace_var, params_data)
        injected = re.sub(
            r'\[resource:(\w+)\]',
            lambda m: str(mcp_resources.get(m.group(1), {}).get('id', '')),
            injected
        )
        return injected
        
    return params_data

