from setuptools import setup

setup(
    name='al_for_design',
    version='0.0.1',
    author='Advaith Narayanan',
    author_email='advaith.narayanan20@gmail.com',
    description='A library for optimally generating data with batch active learning in design contexts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=['AL_for_design'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'sobol_seq',
        'autogluon',
        'Pillow==6.2.1'
    ]
) 