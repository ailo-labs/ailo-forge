# docker_generator.py

def generate_dockerfile(base_model: str) -> str:
    """
    Generate a Dockerfile to run the custom LLM.
    """
    dockerfile_content = f"""# Dockerfile for custom LLM based on {base_model}
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
"""
    return dockerfile_content

