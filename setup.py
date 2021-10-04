from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='mereli',
   version='0.0.1',
   description='Multi-Environment Robotics simulator with Evolution and LearnIng implementations (MERELI).',
   author='Rafael Sendra and Álvaro Gutiérrez',
   author_email='',
   packages=find_packages(),
   install_requires=required, #requirements
)