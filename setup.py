from setuptools import setup

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'matplotlib',
    'obspy']

setup(
    name="mess2014",
    version="0.1.0",
    description="Helper scripts for MESS 2014 workshop.",
    author="Joachim Wassermann, Lion Krischer, Tobias Megies",
    url="https://github.com/obspy/mess2014",
    download_url="https://github.com/obspy/mess2014.git",
    install_requires=INSTALL_REQUIRES,
    keywords=["ObsPy", "Seismology", "MESS"],
    packages=["mess2014"],
    entry_points={},
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Library or " +
        "Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
)
