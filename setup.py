import re
from setuptools import setup, find_packages


# Read property from project's package init file
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)


setup(
    name='ga',
    version=get_property('__version__', 'ga'),
    description='Genetic algorithm package',
    author='SErAphLi',
    url='https://github.com/Seraphli/ga.git',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm'
    ]
)
