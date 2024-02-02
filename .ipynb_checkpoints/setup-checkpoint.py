"""Setup for fanpy.

See:
https://packaging.python.org/en/latest/distributing.html
Templated from:
https://github.com/pypa/sampleproject
"""

from os import path

from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fanpy",
    version="1.0.0",
    description="A module for evaluating, differentiating, and integrating Gaussian functions.",
    long_description=long_description,
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    # This should be your name or the name of the organization which owns the
    # project.
    author="Taewon D. Kim",
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="david.kim.91@gmail.com",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        # How mature is this project? Common values are
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 1 - Planning",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        # Pick your license as you wish
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        # Specify the Python versions you support here.
        # These classifiers are *not* checked by 'pip install'. See instead
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="wavefunction hamiltonian optimization schrodinger equation quantum chemistry",
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.4",
    install_requires=["numpy", "scipy", "cma"],
    extras_require={
        "dev": [
            "tox",
            "pytest",
            "pytest-cov",
            "flake8",
            "flake8-pydocstyle",
            "flake8-import-order",
            "pep8-naming",
            "pylint",
            "bandit",
            "black",
        ],
        "test": ["tox", "pytest", "pytest-cov"],
        "horton": ["horton"],
        "pyscf": ["pyscf"],
        "tensorflow": ["tensorflow"],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={},
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    entry_points={
        "console_scripts": [
            "fanpy_make_script=fanpy.scripts.make_script:main",
            "fanpy_run_calc=fanpy.scripts.run_calc:main",
            "fanpy_make_fanci_script=fanpy.scripts.make_fanci_script:main",
        ]
    },
    # List additional URLs that are relevant to your project as a dict.
    project_urls={
        "Bug Reports": "https://github.com/theochem/gbasis/issues",
        "Organization": "https://github.com/quantumelephant/",
        "Source": "https://github.com/theochem/gbasis/",
    },
    zip_safe=False,
    ext_modules= cythonize(
            [
        # Extension(
        #     "fanpy.objective.schrodinger.cext",
        #     ["fanpy/objective/schrodinger/cext.pyx"],
        #     include_dirs=[numpy.get_include()],
        # ),
        # Extension(
        #     "fanpy.wfn.geminal.cext",
        #     ["fanpy/wfn/geminal/cext.pyx"],
        #     include_dirs=[numpy.get_include()],
        # ),
        Extension(
            "fanpy.upgrades.cext_apg",
            ["fanpy/upgrades/cext_apg.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "fanpy.upgrades.cext_sign",
            ["fanpy/upgrades/cext_sign.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "fanpy.upgrades.cext_objective",
            ["fanpy/upgrades/cext_objective.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "fanpy.upgrades.cext_apg_parallel",
            ["fanpy/upgrades/cext_apg_parallel.pyx"],
            #extra_compile_args=['-fopenmp'],
            #extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "fanpy.upgrades.cext_apg_parallel2",
            ["fanpy/upgrades/cext_apg_parallel2.pyx"],
            #extra_compile_args=['-fopenmp'],
            #extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()],
        ),
        ]
    ),
)
