# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM history compatibility, preserving malformed arguments verbatim."""

_OBJECT_ARGS = (
    "{% set _args = tc.arguments %}"
    "{% for k, v in _args.items() %}<arg_key>{{ k }}</arg_key>"
    "<arg_value>{{ v | tojson(ensure_ascii=False) if v is not string else v }}"
    "</arg_value>{% endfor %}"
)
_RAW_OR_OBJECT_ARGS = (
    "{% set _args = tc.arguments %}"
    '{% if _args is string %}{{ "\\n" + _args }}{% else %}'
    + _OBJECT_ARGS.removeprefix("{% set _args = tc.arguments %}")
    + "{% endif %}"
)


def adapt_glm_history_template(template: str) -> str:
    """Adapt the verified GLM-5.3 template; fail closed on unknown revisions.

    No model files are modified. The resulting template is shared with both
    the inline tokenizer and preprocessing subprocesses. Object rendering is
    unchanged; invalid JSON remains visible to the model as historical text.
    """
    if not isinstance(template, str):
        raise ValueError("GLM history compatibility requires a single Jinja template")
    if template.count(_RAW_OR_OBJECT_ARGS) == 1:
        return template
    if template.count(_OBJECT_ARGS) != 1 or "<tool_call>" not in template:
        raise ValueError(
            "Unrecognized GLM history template; review its argument rendering "
            "before enabling --glm-history-arguments-models"
        )
    return template.replace(_OBJECT_ARGS, _RAW_OR_OBJECT_ARGS, 1)
