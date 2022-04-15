import setuptools

from pysamoo.version import __version__

# ---------------------------------------------------------------------------------------------------------
# GENERAL
# ---------------------------------------------------------------------------------------------------------


__name__ = "pysamoo"
__author__ = "Julian Blank"
__url__ = "https://anyoptimization.com/projects/pysamoo/"

data = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>=3.7',
    author_email="blankjul@msu.edu",
    description="Surrogate-Assisted Multi-objective Optimization",
    license='GNU AFFERO GENERAL PUBLIC LICENSE (AGPL)',
    keywords="surrogate, metamodel, bayesian optimization",
    install_requires=["pymoo>0.5.0", "ezmodel"],
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
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


def packages():
    return ["pysamoo"] + ["pysamoo." + e for e in setuptools.find_packages(where='pysamoo')]


data['long_description'] = readme()
data['packages'] = packages()

setuptools.setup(**data)
