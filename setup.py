from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='mereli',
   version='0.0.1',
   description='Evolutionary Robotics Simulator',
   author='Rafael Sendra',
   author_email='',
   packages=find_packages(),
   install_requires=required, #requirements
)