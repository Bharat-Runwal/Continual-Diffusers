from setuptools import setup, find_packages


# Function to parse requirements from requirements.txt
def load_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()
    
setup(
    name='continual_diffusers',
    version='0.1.0',
    packages=find_packages(),
    description='A library for continual learning with diffusion models',
    author='Bharat Runwal',
    author_email='bharatrunwal@gmail.com',
    install_requires=load_requirements('requirements.txt'),
    python_requires='>=3.7',
)