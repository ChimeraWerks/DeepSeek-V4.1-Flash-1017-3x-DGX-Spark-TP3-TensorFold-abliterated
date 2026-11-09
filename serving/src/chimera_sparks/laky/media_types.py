# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
# Adapted for the recipe-owned TensorFold API; see recipe/UPSTREAM.json.
"""Image block tags accepted by the gateway's Responses normalizer, and the PDF document tags the engine renders
into page images (TensorFold --vision-pdf): Messages document, Chat file, Responses input_file."""
IMAGE_BLOCK_TYPES = frozenset({'input_image', 'image_url', 'image', 'computer_screenshot'})
DOCUMENT_BLOCK_TYPES = frozenset({'document', 'file', 'input_file'})
