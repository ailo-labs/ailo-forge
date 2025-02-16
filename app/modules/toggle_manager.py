def process_toggles(toggles: dict) -> dict:
    """
    Process toggle flags and map them to generation parameters.
    Expected toggles include:
    - "jailbreak": bool
    - "creative_burst": bool
    - "factual_rigor": bool
    - "verbose": bool (True for verbose explanations, False for concise)
    - "domain_focus": str (e.g., "medical", "finance", etc.)
    
    Returns a standardized configuration dict.
    """
    config = {
        "jailbreak": toggles.get("jailbreak", False),
        "creative_burst": toggles.get("creative_burst", False),
        "factual_rigor": toggles.get("factual_rigor", True),
        "verbose": toggles.get("verbose", False),
        "domain_focus": toggles.get("domain_focus", "general")
    }
    return config

