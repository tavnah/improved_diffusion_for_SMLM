from setuptools import setup

setup(
    name="improved_diffusion_for_SMLM",
    py_modules=["improved_diffusion_for_SMLM"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)