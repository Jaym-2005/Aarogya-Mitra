# Aarogya-Mitra



<!-- import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    " test.py"
] 

for filepath in list_of_files:
     filepath = Path(filepath)
     filedir, filename = os.path.split(filepath)

     if filedir !="":
      os.makedirs(filedir, exist_ok=True)
      logging.info(f"Creating directory; {filedir} for the file: {filename}")


     if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
           pass
           logging.info(f"Creating empty file: {filepath}")


     else:
           logging.info(f"{filename} is already exists")       -->


<!-- from setuptools import find_packages, setup

setup(
    name = 'Generative AI Project',
    version= '0.0.0',
    author= 'Light',
    author_email= 'jaymahajan7595@gmail.com',
    packages= find_packages(),
    install_requires = []

) -->