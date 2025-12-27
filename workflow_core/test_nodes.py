from workflow_core.nodes import resolve_template_variables


def test_resolve_template_variables_handles_loop_item_reference():
    variable_map = {
        "for_each_email.item": {
            "subject": "Re: Collaboration",
            "from_email": "founder@example.com",
        },
        "for_each_email.output": {
            "subject": "Re: Collaboration",
            "from_email": "founder@example.com",
        },
    }
    
    template = "Drafting reply to {{for_each_email.item.from_email}}"
    resolved = resolve_template_variables(template, variable_map)
    
    assert resolved == "Drafting reply to founder@example.com"


def test_resolve_template_variables_supports_nested_output_handles():
    variable_map = {
        "summarize_draft_reply.output": {
            "summary_bullets": ["One", "Two"],
        }
    }
    
    template = "Summary: {{summarize_draft_reply.output.summary_bullets.0}}"
    resolved = resolve_template_variables(template, variable_map)
    
    assert resolved == "Summary: One"

