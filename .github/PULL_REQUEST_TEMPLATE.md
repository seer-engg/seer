# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement

## Testing
<!-- Describe the tests you ran and how to reproduce them -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Code Quality Checklist
- [ ] Pylint score maintained or improved (run `pylint api/ shared/ agents/ workflow_compiler/ worker/`)
- [ ] No new `pylint: disable` without architectural justification in code comments
- [ ] Exception handling catches specific exception types (avoid bare `except Exception`)
- [ ] No dynamic attribute attachment to ORM/SQLAlchemy models
- [ ] New disables documented below with reasoning

<!-- If you added new pylint disables, explain the architectural reason here -->
**New Pylint Disables:**
<!--
Example:
- `file.py:123` - import-outside-toplevel - Avoids circular import with services module
-->

## Related Issues
<!-- Link related issues here -->
Closes #
Related to #

## Additional Notes
<!-- Any additional information that reviewers should know -->
