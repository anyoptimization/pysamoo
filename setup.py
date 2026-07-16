import os

import setuptools

# Read the version without importing the package (src/ layout: package not on
# sys.path at build time).
__version__ = {}
with open(os.path.join("src", "pysamoo", "version.py")) as f:
    exec(f.read(), __version__)
__version__ = __version__["__version__"]

# ---------------------------------------------------------------------------------------------------------
# GENERAL
# ---------------------------------------------------------------------------------------------------------


name = "pysamoo"
author = "Julian Blank"
url = "https://anyoptimization.com/projects/pysamoo/"

data = dict(
    name=name,
    version=__version__,
    author=author,
    url=url,
    python_requires='>=3.10',
    author_email="blankjul@msu.edu",
    description="Surrogate-Assisted Multi-objective Optimization",
    license='PolyForm Noncommercial License 1.0.0',
    keywords="surrogate, metamodel, bayesian optimization",
    install_requires=["pymoo>=0.6.1.5,<0.6.2", "ezmodel"],
    extras_require={
        "dev": ["ruff", "mypy", "pytest", "pytest-xdist", "pytest-cov"],
    },
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: Other/Proprietary License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)


# ---------------------------------------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------------------------------------


# update the readme.rst to be part of setup
def readme():
    with open('README.rst') as f:
        return f.read()


data['long_description'] = readme()
data['package_dir'] = {'': 'src'}
data['packages'] = setuptools.find_packages(where='src')

setuptools.setup(**data)
