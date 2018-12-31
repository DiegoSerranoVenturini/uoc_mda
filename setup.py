from setuptools import setup, find_packages

setup(
    name='uocmda',
    version="0.0.1",
    author='Diego Serrano',
    author_email='diego.serrano.venturini@gmail.com',
    url='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='',
    long_description='',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)

