

COMMMON_TOOL_INSTRUCTIONS = """

# Important:
- Asana expects no offset parameter at all on the first page. Sending offset="" (empty string) is treated as an invalid pagination token, so Asana returns:
    offset: Your pagination token is invalid.

- When creating a github repository, do not pass 'team_id' parameter.
"""