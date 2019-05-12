from distutils.core import setup

setup(
    name='evolvegcbh',
    version='0.1.0',
    author='Mark Gieles',
    author_email='mgieles@gmail.com',
    packages=['evolvegcbh'],
    scripts=['evolvegcbh/evolvegcbh.py'],
    url='http://pypi.python.org/pypi/evolvegcbh',
    license='LICENSE.txt',
    description='Fast model for GC with BHs',
    long_description=open('README.md').read(),
    install_requires=[
        "scipy",
        "numpy",
    ],
)
