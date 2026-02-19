# builder.py

from textwrap import dedent
from base import BASE_INSTRUCTION
from tasks import TASK_CONFIGS


# ==========================================================
# Helpers
# ==========================================================

def _format_labels(labels):
    return "\n- " + "\n- ".join(labels)


def _validate_task(task_name):
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Task '{task_name}' not found in TASK_CONFIGS.")


def _validate_support_labels(task_name, support_examples):
    allowed = set(TASK_CONFIGS[task_name]["labels"])
    for ex in support_examples:
        if ex["label"] not in allowed:
            raise ValueError(
                f"Support label '{ex['label']}' not allowed for task '{task_name}'."
            )


def _build_hierarchy_block(task_config, include_hierarchy=True):
    if not include_hierarchy:
        return ""

    hierarchy = task_config.get("hierarchy", "")
    if not hierarchy:
        return ""

    return f"""
----------------------------------------
HIERARCHY REFERENCE:
{hierarchy}
"""


def _build_few_shot_block(support_examples):
    if not support_examples:
        return ""

    block = "\n----------------------------------------\nFEW-SHOT EXAMPLES:\n"

    for i, ex in enumerate(support_examples, 1):
        block += f"""
Example {i}:
Observation:
{ex['observation']}

Description:
{ex['description']}

Label:
{ex['label']}
"""

    return block


def _build_output_instruction(labels):
    labels_text = _format_labels(labels)

    return f"""
----------------------------------------
ALLOWED LABELS:
{labels_text}

Respond strictly in the following format:

Thought: <brief structural reasoning>
Action: <one label from the allowed set above>
"""


# ==========================================================
# Main Builder
# ==========================================================

def build_prompt(
    task_name: str,
    query_observation: str,
    support_examples: list = None,
    include_hierarchy: bool = True
) -> str:
    """
    Build a full classification prompt.

    Parameters:
        task_name (str): Name of task in TASK_CONFIGS
        query_observation (str): The observation to classify
        support_examples (list): List of dicts with keys:
            {
                "observation": "...",
                "description": "...",
                "label": "..."
            }
        include_hierarchy (bool): Whether to include hierarchy reference

    Returns:
        str: Full assembled prompt
    """

    _validate_task(task_name)

    if support_examples is None:
        support_examples = []

    _validate_support_labels(task_name, support_examples)

    task_config = TASK_CONFIGS[task_name]

    base_block = BASE_INSTRUCTION

    task_block = f"""
----------------------------------------
TASK DESCRIPTION:
{task_config['description']}
"""

    hierarchy_block = _build_hierarchy_block(
        task_config,
        include_hierarchy=include_hierarchy
    )

    few_shot_block = _build_few_shot_block(support_examples)

    output_block = _build_output_instruction(task_config["labels"])

    query_block = f"""
----------------------------------------
Now classify the following observation.

Observation:
{query_observation}
"""

    prompt = dedent(f"""
{base_block}
{task_block}
{hierarchy_block}
{few_shot_block}
{query_block}
{output_block}
""").strip()

    return prompt
