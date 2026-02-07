# Seer Workflow System - Unintuitive Aspects

This document highlights design patterns and limitations in the Seer workflow system that may be surprising or counterintuitive to developers, especially those familiar with traditional programming languages.

---

## 1. Type Inference Limitation in Loop Variables

### The Problem

**Loop variables lose their type information**, even when the source data has an explicitly defined JSON schema.

### What Developers Expect

```json
{
  "id": "prepare_data",
  "type": "llm",
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "contacts": {"type": "array"}
          }
        }
      }
    }
  }
}
```

Then in a loop:
```json
{
  "id": "loop_orgs",
  "type": "for_each",
  "items": "${prepare_data}",
  "item_var": "org"
}
```

**Developers expect:** `${org.name}` and `${org.contacts}` to work, since we defined the schema.

### What Actually Happens

**Validation Error:**
```
Property 'name' not declared in schema
Property 'contacts' not declared in schema
```

The loop variable `org` receives a "permissive object schema" at compile time, which paradoxically means it **can't access its own properties**.

### Why This is Unintuitive

1. **Schema was explicit** - We defined exactly what properties exist
2. **Works differently than all major languages** - Python, JavaScript, TypeScript, etc. all preserve type information through iterations
3. **Contradictory naming** - "Permissive" suggests it allows access, but it actually restricts it

### Required Workaround

Use an LLM block to "extract" data that already exists:

```json
{
  "id": "extract_org_data",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Extract from this organization data: ${org}. Return JSON with 'name' and 'contacts' fields."
  },
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "contacts": {"type": "array"}
        }
      }
    }
  }
}
```

### Impact

- **Performance cost**: Extra LLM calls for data that's already structured
- **Latency increase**: Additional round-trips for each loop iteration
- **Cost increase**: LLM tokens consumed just to access existing data
- **Complexity**: Adds extra nodes to every nested loop workflow

---

## 2. Template Expressions vs Condition Expressions

### The Problem

**Different expression syntaxes have dramatically different capabilities**, but use identical `${...}` syntax.

### Template Expressions (in tool inputs, LLM prompts)

**Limited to simple substitution:**

✅ **Allowed:**
```json
"${node_id}"
"${node_id.field}"
"${array[0]}"
```

❌ **NOT Allowed:**
```json
"${index + 1}"          // Arithmetic - FAILS
"${first + ' ' + last}" // Concatenation - FAILS
"${count > 0}"          // Comparison - FAILS
```

### Condition Expressions (in `if` nodes)

**Full expression evaluation:**

✅ **Allowed:**
```json
"${count} + 1 > 5"
"${price} * ${quantity} > 1000"
"len(${items}) > 0"
"${status} == 'active' && ${count} >= 10"
```

### Why This is Unintuitive

1. **Same syntax, different capabilities** - Both use `${...}` but behave completely differently
2. **No visual distinction** - Nothing indicates which expression type you're writing
3. **Common operations fail** - Simple arithmetic like `${index + 1}` doesn't work in templates
4. **Inconsistent with most languages** - Most template systems support basic expressions

### Example: Can't Calculate Row Numbers

**What doesn't work:**
```json
{
  "id": "send_email",
  "type": "tool",
  "tool": "gmail_send_email",
  "inputs": {
    "subject": "You are number ${index + 1}"  // FAILS
  }
}
```

**Required workaround:**
```json
{
  "id": "calculate_row",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Calculate and return only the number: ${index} + 1"
  },
  "outputs": {"mode": "text"}
}
```

Then use `${calculate_row}` in subsequent nodes.

### Impact

- **Unnecessary LLM calls** for simple arithmetic
- **Workflow bloat** - extra nodes for basic operations
- **Debugging confusion** - expressions fail silently or with unclear errors

---

## 3. Array Indexing Limitations

### The Problem

**Array indexing may or may not work** depending on whether the compiler "knows" the value is an array.

### When It Fails

```json
{
  "id": "loop_rows",
  "type": "for_each",
  "items": "${spreadsheet_rows}",  // Each item is ["col1", "col2", "col3"]
  "item_var": "row"
}
```

Then trying to access columns:
```json
{
  "inputs": {
    "name": "${row[0]}",     // May FAIL
    "email": "${row[1]}",    // May FAIL
    "status": "${row[2]}"    // May FAIL
  }
}
```

### Why This is Unintuitive

1. **Works in condition expressions** - `${row[0]}` works in `if` conditions but not templates
2. **Inconsistent behavior** - Sometimes works, sometimes doesn't, based on type inference
3. **Common data pattern** - CSV/spreadsheet data naturally comes as arrays of arrays

### Recommended Workaround

**Design data as objects instead:**
```json
[
  {"name": "Alice", "email": "alice@example.com", "status": "active"},
  {"name": "Bob", "email": "bob@example.com", "status": "inactive"}
]
```

Instead of arrays:
```json
[
  ["Alice", "alice@example.com", "active"],
  ["Bob", "bob@example.com", "inactive"]
]
```

### Impact

- **Data structure constraints** - Forces object-based design even when arrays are more natural
- **Extra transformation** - May need to convert array data to objects before looping

---

## 4. Version String Must Be Exactly "2"

### The Problem

The workflow version must be the string `"2"`, not `"1.0"`, `"2.0"`, `"v2"`, or integer `2`.

### What Developers Might Try

```json
{"version": "2.0"}    // FAILS
{"version": "v2"}     // FAILS
{"version": 2}        // FAILS (integer)
{"version": "1.0"}    // FAILS
```

### Only Valid Format

```json
{"version": "2"}      // ✅ CORRECT
```

### Why This is Unintuitive

1. **Semantic versioning expectation** - Most systems use `"2.0"`, `"2.1"` for versions
2. **Type confusion** - String, not number, but looks like a number
3. **No flexibility** - Strict exact match requirement

### Impact

- **Minor annoyance** - Easy to fix once you know, but catches everyone once
- **Validation errors** - Cryptic errors if you use wrong format

---

## 5. Loop Exit Target Must Be a Real Node

### The Problem

The `loop_exit` edge **must point to an actual node**, not a symbolic target like `"done"` or `"end"`.

### What Doesn't Work

```json
{
  "edges": [
    {
      "source": "loop_id",
      "target": "done",        // FAILS - "done" is not a real node
      "type": "loop_exit"
    }
  ]
}
```

### What You Must Do

Create an actual node, even if it does nothing important:

```json
{
  "nodes": [
    {
      "id": "completion_message",
      "type": "llm",
      "inputs": {
        "model": "gpt-5-mini",
        "prompt": "Loop completed"
      },
      "outputs": {"mode": "text"}
    }
  ],
  "edges": [
    {
      "source": "loop_id",
      "target": "completion_message",  // ✅ Real node
      "type": "loop_exit"
    }
  ]
}
```

### Why This is Unintuitive

1. **Other systems allow symbolic exits** - Many workflow systems have "end" nodes
2. **Forces dummy nodes** - May need to create nodes just to satisfy the requirement
3. **Not documented prominently** - Easy to miss in examples

### Impact

- **Workflow clutter** - Extra nodes that serve no functional purpose
- **Validation errors** - Confusing error messages about undefined targets

---

## 6. Tool Blocks Cannot Have `outputs` Field

### The Problem

Tool blocks **must not** define an `outputs` field, even though LLM blocks require it.

### What Fails

```json
{
  "id": "send_email",
  "type": "tool",
  "tool": "gmail_send_email",
  "inputs": {...},
  "outputs": {          // ❌ ERROR: Tool nodes can't have outputs field
    "mode": "json",
    "schema": {...}
  }
}
```

### Correct Format

```json
{
  "id": "send_email",
  "type": "tool",
  "tool": "gmail_send_email",
  "inputs": {...}
  // No outputs field - schema comes from tool registry
}
```

### Why This is Unintuitive

1. **Inconsistent with LLM blocks** - LLM blocks require `outputs`, tool blocks forbid it
2. **Not obvious from examples** - Easy to copy-paste and add outputs
3. **Implicit schema source** - Schema comes from registry, not definition

### Impact

- **Copy-paste errors** - Easy to make when adapting LLM blocks to tool blocks
- **Validation confusion** - Error message may not be immediately clear

---

## 7. No Arithmetic or String Operations in Templates

### The Problem

Common operations that work in almost every template language **don't work** in Seer template expressions.

### Operations That Don't Work

```json
// Arithmetic
"Row ${index + 1}"                    // FAILS
"Total: ${price * quantity}"          // FAILS
"Discount: ${total * 0.1}"           // FAILS

// String concatenation
"${first_name} ${last_name}"          // FAILS
"Hello, ${title}. ${name}"            // FAILS

// String methods
"${email.toLowerCase()}"              // FAILS
"${text.substring(0, 10)}"           // FAILS

// Array length
"Found ${items.length} items"         // FAILS
"Processing ${len(items)} records"    // FAILS
```

### Comparison to Other Systems

**Jinja2 (Python):**
```jinja
{{ index + 1 }}
{{ first_name + " " + last_name }}
{{ items|length }}
```

**Handlebars (JavaScript):**
```handlebars
{{ add index 1 }}
{{ concat first_name " " last_name }}
```

**Liquid (Ruby):**
```liquid
{{ index | plus: 1 }}
{{ first_name | append: " " | append: last_name }}
```

**All of these work in their respective template systems**, but Seer only supports direct substitution.

### Required Workaround

Use LLM blocks for every calculation:

```json
{
  "id": "calculate_total",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Calculate: ${price} * ${quantity}. Return only the number."
  },
  "outputs": {"mode": "text"}
}
```

### Impact

- **High LLM usage** for trivial operations
- **Increased latency** - multiple round-trips for simple calculations
- **Cost increase** - LLM tokens for arithmetic
- **Workflow complexity** - many extra nodes

---

## 8. Secrets Only Work in MCP Blocks

### The Problem

The `${secrets.key}` syntax **only works in MCP blocks**, not in regular tool blocks.

### Where It Works

```json
{
  "type": "mcp",
  "server": "http://api.example.com",
  "auth": {
    "headers": {
      "Authorization": "Bearer ${secrets.api_key}"  // ✅ Works
    }
  }
}
```

### Where It Doesn't Work

```json
{
  "type": "tool",
  "tool": "custom_api_call",
  "inputs": {
    "api_key": "${secrets.api_key}"  // ❌ Doesn't work (probably)
  }
}
```

### Why This is Unintuitive

1. **Inconsistent availability** - Same syntax, different support
2. **Security pattern** - Secrets should be usable everywhere sensitive data is needed
3. **Not clearly documented** - The limitation isn't prominently stated

### Impact

- **Limited secret management** - Can't use secrets with regular tools
- **Workaround complexity** - May need MCP wrappers just to use secrets

---

## Summary: Impact on Developer Experience

### High-Impact Issues

1. **Type inference in loops** - Forces workarounds on every nested loop
2. **No arithmetic in templates** - Requires LLM calls for trivial operations
3. **Template vs condition expression split** - Hard to remember which supports what

### Medium-Impact Issues

4. **Array indexing limitations** - Forces specific data structure patterns
5. **Loop exit requires real node** - Creates workflow clutter
6. **Tool blocks can't have outputs** - Easy to get wrong initially

### Low-Impact Issues

7. **Version must be "2"** - Annoying once, then you remember
8. **Secrets only in MCP blocks** - Workaround exists

---

## Recommendations for Improvement

### For Documentation

1. **Add prominent warnings** before examples that use nested loops
2. **Show the workaround first** - Don't let developers hit the error
3. **Create a "Common Mistakes" section** in the guide
4. **Add inline examples** of what doesn't work with explanations

### For System Design

1. **Improve type inference** - Preserve schema information through loops
2. **Add basic template expressions** - At minimum: `+`, `-`, `*`, `/`, `concat`
3. **Support arithmetic in templates** - Most template systems have this
4. **Allow symbolic loop exits** - Or auto-create completion nodes
5. **Unify expression syntax** - Same capabilities everywhere, or clearly distinguish syntax

### For Error Messages

1. **Detect type inference issues** - Suggest the LLM extraction workaround
2. **Better validation messages** - "Consider using an LLM block to extract properties from loop variables"
3. **Detect arithmetic in templates** - "Arithmetic not supported in template expressions. Use a condition expression or LLM block."

---

## Conclusion

The Seer workflow system is powerful, but several design choices create friction for developers:

- **Type safety is enforced inconsistently** - Strict in loops, permissive elsewhere
- **Expression capabilities vary by context** - Same syntax, different features
- **Workarounds require LLM calls** - Simple operations need full AI inference
- **Documentation doesn't emphasize limitations** - Easy to hit errors before reading warnings

These issues are **documented but not prominent**, leading to trial-and-error development even when reading the guide first.

**The good news:** Once you know the patterns, they're manageable. This document should help new developers avoid the common pitfalls.
