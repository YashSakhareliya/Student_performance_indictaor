from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def find_requirements(file_path:str)->list[str]:
    '''
        This Function will return List of requiremente(depedencyes)
    '''
    requiremets = []
    with open(file_path,'r') as file:
        requiremets=file.readlines()
        requiremets=[req.replace('\n','') for req in requiremets]
        if HYPEN_E_DOT in requiremets:
            requiremets.remove(HYPEN_E_DOT)
    
    return requiremets

setup(
    name='ml_project',
    version='0.0.1',
    author='Yash',
    author_email='sakhareliyayash@gmail.com',
    packages=find_packages(),
    install_requires=find_requirements('requirements.txt')

)