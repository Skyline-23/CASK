from setuptools import setup, find_packages

setup(
    name="cask",
    version="0.1.0",
    description="CASK: Core-Aware Selective KV Compression for Reasoning Traces",
    author="CASK contributors",
    url="https://github.com/Skyline-23/CASK",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.48.1",
        "datasets>=4.0",
        "huggingface-hub>=0.35",
        "accelerate",
        "numpy>=1.26",
        "scipy",
        "einops",
        "sentencepiece",
        "pyyaml>=6.0",
        "tqdm",
        "matplotlib",
        "regex",
        "jieba",
        "fuzzywuzzy",
        "python-Levenshtein",
        "rouge",
        "nltk",
        "wonderwords",
        "tenacity",
        "torch",
        "triton",
    ],
    extras_require={
        "eval": [
            "pebble>=5.0",
            "sympy>=1.13",
            "latex2sympy2",
            "word2number",
            "timeout-decorator",
            "antlr4-python3-runtime==4.11.1",
        ],
        "flash": ["flash-attn>=2.5.8"],
    },
    entry_points={
        "vllm.general_plugins": [
            "cask = cask.vllm.plugin:register_triattention_backend",
        ],
    },
)


