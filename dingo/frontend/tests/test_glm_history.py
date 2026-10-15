# SPDX-License-Identifier: Apache-2.0
"""GLM history repair must preserve valid prompts and the original invalid text."""

import json

import pytest
from jinja2 import Environment, StrictUndefined

from dingo.frontend.glm_history import adapt_glm_history_template

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# Representative GLM-5.3 tool-call block, independent of the adapter constants.
TEMPLATE = (
    "{{ '<tool_call>' + tc.name }}"
    "{% set _args = tc.arguments %}{% for k, v in _args.items() %}"
    "<arg_key>{{ k }}</arg_key><arg_value>"
    "{{ v | tojson(ensure_ascii=False) if v is not string else v }}"
    "</arg_value>{% endfor %}</tool_call>"
)


def render(template, arguments):
    env = Environment(undefined=StrictUndefined, autoescape=False)
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, **kw)
    return env.from_string(template).render(tc={"name": "lookup", "arguments": arguments})


@pytest.mark.parametrize("arguments", [{}, {"city": "北京"}, {"a": [1, 2], "b": {"x": True}}])
def test_valid_object_prompt_unchanged(arguments):
    assert render(adapt_glm_history_template(TEMPLATE), arguments) == render(TEMPLATE, arguments)


@pytest.mark.parametrize("arguments", ['{"city":', '{"a": 1,}', 'not json', '{"x":"{{ injected }}'])
def test_invalid_history_preserved_verbatim(arguments):
    assert render(adapt_glm_history_template(TEMPLATE), arguments) == (
        "<tool_call>lookup\n" + arguments + "</tool_call>"
    )


def test_idempotent_and_unknown_template_rejected():
    adapted = adapt_glm_history_template(TEMPLATE)
    assert adapt_glm_history_template(adapted) == adapted
    for unknown in (None, {}, "{{ messages }}", TEMPLATE + TEMPLATE):
        with pytest.raises(ValueError):
            adapt_glm_history_template(unknown)
