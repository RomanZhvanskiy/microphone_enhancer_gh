from setuptools import find_packages
from setuptools import setup

with open("requirements_prod.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='microphone-enhancer',
      version="0.0.4.1",
      description="microphone enhancer backend",
      license="MIT",
      author="",
      #author_email="rz03@outlook.com",
      #url="https://github.com/RomanZhvanskiy/microphone_enhancer_gh/",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
