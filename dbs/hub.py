"""Hugging Face Hub integration shared by every released model."""
from __future__ import annotations

from huggingface_hub import PyTorchModelHubMixin

REPO_URL = "https://github.com/JusperLee/Dynamic-Band-Splitting"
PROJECT_URL = "https://cslikai.cn/Dynamic-Band-Splitting/"

HUB_KWARGS = dict(
    repo_url=REPO_URL,
    docs_url=PROJECT_URL,
    license="apache-2.0",
    pipeline_tag="audio-to-audio",
    tags=["audio-source-separation", "dynamic-band-splitting"],
)

__all__ = ["HUB_KWARGS", "PyTorchModelHubMixin", "REPO_URL", "PROJECT_URL"]
