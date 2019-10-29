from distutils.core import setup

setup(
    name='clusterbh',
    version='0.1.0',
    author='Mark Gieles',
    author_email='mgieles@gmail.com',
    packages=['clusterbh'],
    scripts=['clusterbh/clusterbh.py'],
    url='http://githb.com/mgieles/clusterbh',
    license='LICENSE.txt',
    description='Fast model for GC with BHs',
    long_description=open('README.md').read(),
    install_requires=[
        "scipy",
        "numpy",
    ],
)
