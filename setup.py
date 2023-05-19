""" Package setup """
from setuptools import find_packages, setup
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup (
    name='vonmises',
    version='0.0.1',
    author='Kirk Swanson',
    author_email='swansonk1@uchicago.edu',
    description='Conformational Search using Deep Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ks8/vonmises',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords=[
        'chemistry',
        'machine learning',
        'neural network',
        'conformation',
        'molecular dynamics'
    ]
)
