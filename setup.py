from distutils.core import setup
setup(
    name = 'prog_models',
    packages = ['prog_models'],
    version = '1.0',
    license = 'NOSA',
    description = 'The NASA Prognostic Model Package is a python modeling framework focused on defining and building models for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components.',
    author = 'Christopher Teubert',
    author_email = 'christopher.a.teubert@nasa.gov',
    url = '', # TBD
    download_url = '', # TBD
    keywords = ['prognostics', 'diagnostics', 'fault detection', 'physics modeling', 'prognostics and health management', 'PHM', 'health management'],
    install_requires = [
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',     
        'Topic :: Research Tools :: Prognostics and Health Management',
        'License :: NASA Open Source Agreement',   
        'Programming Language :: Python :: 3',     
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)