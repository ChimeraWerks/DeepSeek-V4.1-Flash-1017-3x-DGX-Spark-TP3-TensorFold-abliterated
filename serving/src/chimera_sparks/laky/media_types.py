# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
# Adapted for the recipe-owned TensorFold API; see recipe/UPSTREAM.json.
"""Image block tags accepted by the gateway's Responses normalizer."""
IMAGE_BLOCK_TYPES = frozenset({'input_image', 'image_url', 'image', 'computer_screenshot'})
