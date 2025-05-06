from setuptools import setup, find_packages

setup(
    name="meformer",
    version="0.1",
    packages=find_packages(include=["projects*", "projects.*"]),
    python_requires=">=3.8",
    install_requires=[
        # core runtime deps
        "mmcv-full>=1.3.17,<=1.6.0",           # must satisfy mmdet3d constraints :contentReference[oaicite:3]{index=3}
        "mmdet==2.24.0",
        "mmdet3d==1.0.0rc5",
        "mmsegmentation==0.29.1",
        "flash-attn==0.2.2",
    ],
    extras_require={
        # GPU‐specific or optional
        "spconv": ["spconv-cu111"],
    },
)
