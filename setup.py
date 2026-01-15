from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name='micrograd',
    version='0.1',
    author='Thomas Brunel',
    author_email='thomas.brunel@hec.edu',
    description="A reproduction of Karpathy's micrograd for educational purpose",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url='https://github.com/thombrunel/my-micrograd',

    packages=find_packages(),
    
    install_requires = [
        "numpy",
    ],

    extras_require = {
        "dev": [
            "graphviz>=0.20",
        ],
    },

    python_requires='>=3.8'
)