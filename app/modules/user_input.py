def parse_input(description: str, toggles: dict) -> dict:
    """
    Parse the natural language description and toggles,
    then standardize into a structured format.
    """
    if not description:
        raise ValueError("Description cannot be empty.")
    # Basic parsing - In practice, add NLP parsing if needed.
    return {"description": description, "toggles": toggles}

