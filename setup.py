# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="tools",
    ext_modules=cythonize(
        module_list=[
            "treat_compare_img.py",
            "simplefem_focus.py",
            "server.py",
            "ocr_detect.py",
            "pynvml.py",
            "image_difference.py",
        ],
        language_level=3,
    ),
)
