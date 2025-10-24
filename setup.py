import setuptools

with open('Readme.md') as fp:
    long_description = fp.read()

with open('requirements.txt') as fp:
    requirements = [
        line.strip()
        for line in fp
        if line.strip() and not line.strip().startswith('#')
    ]

setuptools.setup(
    name="preprocess",
    include_package_data=True,
    version="0.1.0",
    author="Kushala Rani Talakad Manjunath",
    author_email="kushalatalakad@gmail.com",
    description="A package for text preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)
