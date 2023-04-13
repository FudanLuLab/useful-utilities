from setuptools import setup

setup(
    name='tmt',
    version='0.1',
    py_modules=['tmt'],
    install_requires=[
        'Click',
        'attrs',
        'numpy',
        'pyteomics'
    ],
    entry_points='''
        [console_scripts]
        tmt=tmt:app
    ''',
)
