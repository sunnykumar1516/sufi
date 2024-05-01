import setuptools

with open("README.md", "r") as fh:
	description = fh.read()

setuptools.setup(
	name="sufi",
	version="1.0.1",
	author="sunny kumar",
	author_email="sunnykumar1516@gmail.com",
	packages=["sufi"],
	description="apply different types of filters on image",
	long_description=description,
	long_description_content_type="text/markdown",
	url="https://github.com/sunnykumar1516/sufi",
	license='MIT',
	python_requires='>=3.8',
	install_requires=[
        'numpy>= 1.2',
        'opencv-python>=4'

    ]
)
