# main.py

import argparse
import os
from src.generators.code_generator import create_custom_llm_code
from src.generators.docker_generator import generate_dockerfile
from src.config import BASE_MODELS

def main():
    parser = argparse.ArgumentParser(description="Ailo Forge™ - Custom LLM Generator")
    parser.add_argument("--description", type=str, required=True, help="Describe the desired custom LLM")
    parser.add_argument("--base_model", type=str, choices=BASE_MODELS.keys(), required=True, help="Choose a base model")
    parser.add_argument("--toggle_unrestricted", action="store_true", help="Enable Unrestricted Mode")
    parser.add_argument("--toggle_creative", action="store_true", help="Enable Creative Burst Mode")
    parser.add_argument("--toggle_factual", action="store_true", help="Enable Factual Rigor Mode")
    parser.add_argument("--toggle_concise", action="store_true", help="Enable Concise Summary Mode")
    parser.add_argument("--domain_focus", type=str, default="", help="Specify domain focus (e.g., Medical, Finance)")
    args = parser.parse_args()

    # Gather toggle settings into a dictionary
    toggles = {
        "unrestricted": args.toggle_unrestricted,
        "creative_burst": args.toggle_creative,
        "factual_rigor": args.toggle_factual,
        "concise_summary": args.toggle_concise,
        "domain_focus": args.domain_focus if args.domain_focus else None
    }

    # Select the chosen base model (for now, a placeholder path is used)
    base_model = args.base_model

    # Generate the custom LLM code
    generated_code = create_custom_llm_code(args.description, toggles, base_model)
    
    # Generate the corresponding Dockerfile
    dockerfile_content = generate_dockerfile(base_model)

    # Create an output directory to store the package
    output_dir = "output_package"
    os.makedirs(output_dir, exist_ok=True)

    # Write the generated code to a file
    with open(os.path.join(output_dir, "custom_llm.py"), "w") as code_file:
        code_file.write(generated_code)
    
    # Write the Dockerfile
    with open(os.path.join(output_dir, "Dockerfile"), "w") as docker_file:
        docker_file.write(dockerfile_content)
    
    print("Custom LLM package generated successfully in the 'output_package' directory.")

if __name__ == "__main__":
    main()

