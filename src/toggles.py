# toggles.py

def apply_toggle_settings(prompt: str, toggles: dict) -> str:
    """
    Modify the prompt based on toggle settings.
    Each toggle adds specific instructions to tailor model behavior.
    """
    if toggles.get("unrestricted", False):
        prompt += "\n[UNRESTRICTED MODE ENABLED]"
    if toggles.get("creative_burst", False):
        prompt += "\n[CREATIVE BURST MODE: maximize imaginative output]"
    if toggles.get("factual_rigor", False):
        prompt += "\n[FACTUAL RIGOR MODE: enforce strict factual accuracy]"
    if toggles.get("concise_summary", False):
        prompt += "\n[CONCISE SUMMARY MODE: provide brief responses]"
    if toggles.get("domain_focus"):
        prompt += f"\n[DOMAIN FOCUS: {toggles['domain_focus']}]"
    # Additional toggles can be added here.
    return prompt

