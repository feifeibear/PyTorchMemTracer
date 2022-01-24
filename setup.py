from setuptools import setup, find_packages

def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


require_list = fetch_requirements("requirements.txt")

setup(
    name="pytorchmemtracer",
    version="0.1.6",
    description="pytorchmemtracer",
    author="feifeibear",
    author_email="fangjiarui123@gmail.com",
    url="https://fangjiarui.github.io/",
    install_requires=require_list,
    setup_requires=require_list,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
