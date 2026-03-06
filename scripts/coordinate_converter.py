"""
Coordinate conversion wrapper for using Qwen3-VL models with UI-TARS desktop integration.

This handles the coordinate system differences between:
- Qwen2.5-VL (UI-TARS): Uses Qwen2_5_VLProcessor + Qwen2VLImageProcessor
- Qwen3-VL: Uses Qwen3VLProcessor + Qwen2VLImageProcessorFast
"""

from typing import Tuple, Dict, Any, Optional
import torch
from PIL import Image


class Qwen3ToUITARSCoordinateConverter:
    """
    Converts coordinates from Qwen3-VL output format to UI-TARS compatible format.

    The main issue is that different processors may:
    1. Use different image resizing strategies
    2. Apply different padding
    3. Use different coordinate normalization ranges
    """

    def __init__(
        self,
        qwen3_processor,
        uitars_processor=None,
        screen_width: int = 1920,
        screen_height: int = 1080,
        debug: bool = True,
    ):
        """
        Args:
            qwen3_processor: Qwen3VLProcessor instance
            uitars_processor: Optional Qwen2_5_VLProcessor for comparison
            screen_width: Target screen width in pixels
            screen_height: Target screen height in pixels
            debug: Print debug information
        """
        self.qwen3_processor = qwen3_processor
        self.uitars_processor = uitars_processor
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.debug = debug

        # Get processor configs to understand coordinate systems
        self._analyze_processors()

    def _analyze_processors(self):
        """Analyze processor configurations to understand coordinate differences."""
        if self.debug:
            print("\n" + "=" * 60)
            print("PROCESSOR ANALYSIS")
            print("=" * 60)

            print("\n### Qwen3-VL Processor ###")
            if hasattr(self.qwen3_processor, "image_processor"):
                img_proc = self.qwen3_processor.image_processor
                print(f"Type: {type(img_proc).__name__}")
                if hasattr(img_proc, "size"):
                    print(f"Size config: {img_proc.size}")
                if hasattr(img_proc, "do_resize"):
                    print(f"Do resize: {img_proc.do_resize}")
                if hasattr(img_proc, "resample"):
                    print(f"Resample: {img_proc.resample}")

            if self.uitars_processor:
                print("\n### UI-TARS Processor ###")
                if hasattr(self.uitars_processor, "image_processor"):
                    img_proc = self.uitars_processor.image_processor
                    print(f"Type: {type(img_proc).__name__}")
                    if hasattr(img_proc, "size"):
                        print(f"Size config: {img_proc.size}")
                    if hasattr(img_proc, "do_resize"):
                        print(f"Do resize: {img_proc.do_resize}")

            print("\n" + "=" * 60 + "\n")

    def get_image_preprocessing_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get information about how the image is preprocessed by Qwen3-VL.

        Args:
            image: PIL Image

        Returns:
            Dictionary with preprocessing information
        """
        original_size = image.size  # (width, height)

        # Process image to see what transformations are applied
        # Note: We're not actually using the processed image, just analyzing it
        dummy_messages = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
        ]

        try:
            processed = self.qwen3_processor.apply_chat_template(
                dummy_messages, add_generation_prompt=False, tokenize=False
            )

            # Try to extract image dimensions from processor
            if hasattr(self.qwen3_processor, "image_processor"):
                img_proc = self.qwen3_processor.image_processor
                processed_size = getattr(img_proc, "size", None)
            else:
                processed_size = None

            info = {
                "original_width": original_size[0],
                "original_height": original_size[1],
                "processed_size": processed_size,
                "aspect_ratio": original_size[0] / original_size[1],
            }

            if self.debug:
                print(f"\nImage Preprocessing Info:")
                print(f"  Original: {original_size[0]}x{original_size[1]}")
                print(f"  Processed: {processed_size}")
                print(f"  Aspect Ratio: {info['aspect_ratio']:.3f}")

            return info

        except Exception as e:
            if self.debug:
                print(f"Warning: Could not analyze preprocessing: {e}")
            return {
                "original_width": original_size[0],
                "original_height": original_size[1],
                "aspect_ratio": original_size[0] / original_size[1],
            }

    def convert_coordinates(
        self,
        x: float,
        y: float,
        original_image: Image.Image,
        coordinate_format: str = "auto",
    ) -> Tuple[int, int]:
        """
        Convert coordinates from Qwen3-VL output to screen pixels.

        Args:
            x: X coordinate from Qwen3-VL output
            y: Y coordinate from Qwen3-VL output
            original_image: The original PIL Image sent to the model
            coordinate_format: "auto", "normalized_0_1", "normalized_0_1000", "pixels"

        Returns:
            (pixel_x, pixel_y) in screen coordinates
        """
        # Get preprocessing info
        img_info = self.get_image_preprocessing_info(original_image)
        orig_w, orig_h = img_info["original_width"], img_info["original_height"]

        # Auto-detect coordinate format if needed
        if coordinate_format == "auto":
            coordinate_format = self._detect_coordinate_format(x, y, orig_w, orig_h)

        if self.debug:
            print(f"\nConverting coordinates:")
            print(f"  Input: ({x}, {y})")
            print(f"  Format: {coordinate_format}")

        # Convert based on detected format
        if coordinate_format == "normalized_0_1":
            # Coordinates are in 0.0-1.0 range
            pixel_x = int(x * self.screen_width)
            pixel_y = int(y * self.screen_height)

        elif coordinate_format == "normalized_0_1000":
            # Coordinates are in 0-1000 range (common in Qwen models)
            pixel_x = int((x / 1000) * self.screen_width)
            pixel_y = int((y / 1000) * self.screen_height)

        elif coordinate_format == "pixels_image":
            # Coordinates are in image pixel space, need to scale to screen
            scale_x = self.screen_width / orig_w
            scale_y = self.screen_height / orig_h
            pixel_x = int(x * scale_x)
            pixel_y = int(y * scale_y)

        elif coordinate_format == "pixels_screen":
            # Already in screen pixels
            pixel_x = int(x)
            pixel_y = int(y)

        else:
            raise ValueError(f"Unknown coordinate format: {coordinate_format}")

        if self.debug:
            print(f"  Output: ({pixel_x}, {pixel_y}) screen pixels")

        return pixel_x, pixel_y

    def _detect_coordinate_format(
        self, x: float, y: float, img_w: int, img_h: int
    ) -> str:
        """Auto-detect the coordinate format based on value ranges."""

        # Check if normalized 0-1
        if 0 <= x <= 1 and 0 <= y <= 1:
            return "normalized_0_1"

        # Check if 0-1000 normalized (common in Qwen models)
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return "normalized_0_1000"

        # Check if image pixels
        if x <= img_w and y <= img_h:
            return "pixels_image"

        # Check if screen pixels
        if x <= self.screen_width and y <= self.screen_height:
            return "pixels_screen"

        # Default to normalized if uncertain
        if x <= 1 and y <= 1:
            return "normalized_0_1"

        return "normalized_0_1000"

    def convert_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        original_image: Image.Image,
        coordinate_format: str = "auto",
    ) -> Tuple[int, int, int, int]:
        """
        Convert bounding box coordinates from Qwen3-VL output to screen pixels.

        Args:
            x1, y1: Top-left corner from Qwen3-VL output
            x2, y2: Bottom-right corner from Qwen3-VL output
            original_image: The original PIL Image sent to the model
            coordinate_format: "auto", "normalized_0_1", "normalized_0_1000", "pixels"

        Returns:
            (pixel_x1, pixel_y1, pixel_x2, pixel_y2) in screen coordinates
        """
        px1, py1 = self.convert_coordinates(x1, y1, original_image, coordinate_format)
        px2, py2 = self.convert_coordinates(x2, y2, original_image, coordinate_format)
        return px1, py1, px2, py2


def create_uitars_compatible_wrapper(
    qwen3_model,
    qwen3_processor,
    screen_width: int = 1920,
    screen_height: int = 1080,
    debug: bool = True,
):
    """
    Create a wrapper that makes Qwen3-VL output compatible with UI-TARS desktop.

    Usage:
        model, processor, converter = create_uitars_compatible_wrapper(
            qwen3_model,
            qwen3_processor,
            screen_width=1920,
            screen_height=1080
        )

        # Use model normally
        output = model.generate(...)

        # Convert coordinates from output
        pixel_x, pixel_y = converter.convert_coordinates(x, y, original_image)
    """
    converter = Qwen3ToUITARSCoordinateConverter(
        qwen3_processor=qwen3_processor,
        screen_width=screen_width,
        screen_height=screen_height,
        debug=debug,
    )

    return qwen3_model, qwen3_processor, converter


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QWEN3-VL TO UI-TARS COORDINATE CONVERTER")
    print("=" * 60)

    print("\nEXAMPLE USAGE:\n")

    print("""
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from coordinate_converter import create_uitars_compatible_wrapper
from PIL import Image
import pyautogui

# Load Qwen3-VL model
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

# Create wrapper with your screen resolution
model, processor, converter = create_uitars_compatible_wrapper(
    model,
    processor,
    screen_width=1920,
    screen_height=1080,
    debug=True
)

# Take screenshot
screenshot = pyautogui.screenshot()

# Get model prediction
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": screenshot},
        {"type": "text", "text": "Click on the browser icon"}
    ]
}]

inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
response = processor.decode(outputs[0], skip_special_tokens=True)

# Parse coordinates from response (example: "Click at [523, 678]")
import re
coords = re.findall(r'\\[(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?)\\]', response)
if coords:
    x, y = float(coords[0][0]), float(coords[0][1])
    
    # Convert to screen pixels
    pixel_x, pixel_y = converter.convert_coordinates(x, y, screenshot)
    
    # Now click with correct coordinates
    pyautogui.click(pixel_x, pixel_y)
    print(f"Clicked at screen position: ({pixel_x}, {pixel_y})")
""")

    print("\n" + "=" * 60)
    print("TESTING WITH MOCK DATA")
    print("=" * 60 + "\n")

    # Mock test
    from PIL import Image
    import numpy as np

    # Create mock image
    mock_image = Image.fromarray(
        np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    )

    # Mock processor class
    class MockProcessor:
        class MockImageProcessor:
            def __init__(self):
                self.size = {"height": 1080, "width": 1920}
                self.do_resize = True

        def __init__(self):
            self.image_processor = self.MockImageProcessor()

    mock_qwen3_processor = MockProcessor()

    # Create converter
    converter = Qwen3ToUITARSCoordinateConverter(
        qwen3_processor=mock_qwen3_processor,
        screen_width=1920,
        screen_height=1080,
        debug=True,
    )

    # Test different coordinate formats
    test_cases = [
        (0.5, 0.5, "normalized_0_1", "Center of screen"),
        (500, 500, "normalized_0_1000", "0-1000 normalized"),
        (960, 540, "pixels_image", "Center in image pixels"),
    ]

    for x, y, fmt, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: ({x}, {y}) in format '{fmt}'")
        px, py = converter.convert_coordinates(x, y, mock_image, coordinate_format=fmt)
        print(f"Output: ({px}, {py}) screen pixels")
        print("-" * 60)
