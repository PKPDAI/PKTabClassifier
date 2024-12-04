from typing import List, Optional


def process_llm_output(answer: str, categories: List[str]) -> Optional[str]:
    """
    Process the output from a generative LLM by checking if any category from the list
    is present in the answer. The function returns the first matching category (case-insensitive).

    Args:
        answer (str): The output answer from the LLM as a string.
        categories (List[str]): A list of category strings to check for in the answer.

    Returns:
        Optional[str]: The first category found in the answer, or None if no category matches.
    """
    # Iterate over each category in the list
    for category in categories:
        # Check if the category is present in the answer (case-insensitive)
        if category.lower() in answer.lower():
            return category  # Return the first matching category

    # Return None if no category is found in the answer
    return None


def replace_values_at_indices(a: List, b: List, indices: List[int]) -> List:
    """
    Replace values in list `a` at the specified `indices` with corresponding values from list `b`.

    Args:
        a (List): The list in which values will be replaced.
        b (List): The list of replacement values.
        indices (List[int]): A list of indices in `a` where the values from `b` will be placed.

    Returns:
        List: The modified list `a` with replaced values at the specified indices.

    Raises:
        ValueError: If the length of `b` does not match the length of `indices`.
    """
    # Ensure the number of replacement values matches the number of indices
    if len(b) != len(indices):
        raise ValueError("The number of replacement values must match the number of indices.")

    # Iterate over indices and corresponding replacement values, and perform replacement
    for i, index in enumerate(indices):
        a[index] = b[i]

    return a
