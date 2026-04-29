"""
kk_utils.agents.coder.adapters — Coder adapter implementations

Available adapters:
- DescImageCoderAdapter: Image description generation via coder CLI
- CsvGeneratorCoderAdapter: CSV generation via coder CLI

Usage:
    from kk_utils.agents.coder.adapters import DescImageCoderAdapter

    adapter = DescImageCoderAdapter()
    response = await adapter.execute_coder(
        prompt_text="Describe the image at images/photo.jpg",
        context={"image_path": "images/photo.jpg"},
    )
"""

from .desc_image.adapter import DescImageCoderAdapter
from .csv_generator.adapter import CsvGeneratorCoderAdapter

__all__ = [
    "DescImageCoderAdapter",
    "CsvGeneratorCoderAdapter",
]
