from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path) as f:
        requirements = f.read().splitlines()
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="generative_ai_project",   # no spaces
    version="0.0.1",
    author="Light",
    author_email="jaymahajan7595@gmail.com",
    description="A Generative AI Project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8",
)
