from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name = 'prinpy',
    version = '0.0.3.1',
    license = "MIT",
    description = "A package for fitting principal curves in Python",
    author = "https://github.com/artusoma/",
    author_email = 'artusoma1@gmail.com',
    url = 'https://github.com/artusoma/prinPy', 
    packages = ["prinpy"],
    long_description = long_description,
    long_description_content_type='text/markdown',
    install_requires = ['numpy',
                        'matplotlib',
                        'scipy',
                        'keras',
                        'tensorflow',
                        ],
)