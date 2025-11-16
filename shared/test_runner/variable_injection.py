"""Variable injection utilities for test runner."""
import re
from typing import Any, Dict


def get_field(obj: Any, field_path: str) -> Any:
    """Helper to get a nested field from an object using dot notation and array indexing.
    
    Examples:
        - 'items.0.name' -> obj['items'][0]['name']
        - 'items[0].name' -> obj['items'][0]['name']
    """
    if not obj or not field_path:
        return obj
    try:
        # Parse path with support for both dot notation and bracket notation
        # Convert 'items[0].owner.login' to ['items', '0', 'owner', 'login']
        parts = re.findall(r'(\w+)|\[(\d+)\]', field_path)
        keys = [match[0] or match[1] for match in parts]
        
        value = obj
        for key in keys:
            if not key:
                continue
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                if idx < len(value):
                    value = value[idx]
                else:
                    return None  # Index out of range
            else:
                return None  # Path is invalid
        return value
    except Exception:
        return None


def inject_variables(
    params_data: Any, # Can be dict, list, or str
    variables: Dict[str, Any], 
    mcp_resources: Dict[str, Any]
) -> Any:
    """
    Recursively injects variables and resource IDs into tool parameters.
    - [var:var_name] is replaced by variables['var_name']
    - [resource:resource_key] is replaced by mcp_resources['resource_key']['id']
    - [resource:resource_key.field] is replaced by a nested field
    """
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
        # 1. Check for full-string replacement
        
        # Inject [var:var_name.path.to.field] or [var:var_name]
        # Updated regex to support dots, brackets, and alphanumeric chars
        var_match = re.match(r"^\[var:([\w\.\[\]]+)\]$", params_data)
        if var_match:
            var_path = var_match.group(1)
            # Split into variable name and field path
            if '.' in var_path or '[' in var_path:
                # Extract base variable name (everything before first . or [)
                base_var = re.match(r'^(\w+)', var_path).group(1)
                field_path = var_path[len(base_var):].lstrip('.')
                base_value = variables.get(base_var)
                if base_value is not None:
                    return get_field(base_value, field_path)
                return None
            else:
                return variables.get(var_path)
        
        # Inject [resource:resource_key] (assumes ID)
        res_match = re.match(r"^\[resource:(\w+)\]$", params_data)
        if res_match:
            res_key = res_match.group(1)
            return mcp_resources.get(res_key, {}).get('id')
            
        # Inject [resource:resource_key.field_name]
        res_field_match = re.match(r"^\[resource:(\w+)\.(.+)\]$", params_data)
        if res_field_match:
            res_key = res_field_match.group(1)
            field_path = res_field_match.group(2)
            return get_field(mcp_resources.get(res_key), field_path)

        # 2. Check for inline (partial string) replacement
        
        # Handle inline [var:...] with path support
        def replace_var(match):
            var_path = match.group(1)
            if '.' in var_path or '[' in var_path:
                # Extract base variable name
                base_var = re.match(r'^(\w+)', var_path).group(1)
                field_path = var_path[len(base_var):].lstrip('.')
                base_value = variables.get(base_var)
                if base_value is not None:
                    result = get_field(base_value, field_path)
                    return str(result) if result is not None else ''
                return ''
            else:
                return str(variables.get(var_path, ''))
        
        injected_str = re.sub(
            r'\[var:([\w\.\[\]]+)\]',
            replace_var,
            params_data
        )
        # Handle inline [resource:...] (assumes ID)
        injected_str = re.sub(
            r'\[resource:(\w+)\]', 
            lambda m: str(mcp_resources.get(m.group(1), {}).get('id', '')), 
            injected_str
        )
        return injected_str
        
    return params_data

