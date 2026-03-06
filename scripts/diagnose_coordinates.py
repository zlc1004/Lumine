"""
Diagnostic script to identify coordinate system differences between
Qwen3-VL-235B and UI-TARS models.

Run this script with both models to compare their coordinate outputs.
"""

import json
from typing import Dict, Tuple


def analyze_coordinate_output(
    model_output: str,
    screen_width: int = 1920,
    screen_height: int = 1080,
    image_width: int = None,
    image_height: int = None,
) -> Dict:
    """
    Analyze coordinate output from a VLM model.

    Args:
        model_output: Raw model output containing coordinates
        screen_width: Actual screen width in pixels
        screen_height: Actual screen height in pixels
        image_width: Width of the image sent to model (if known)
        image_height: Height of the image sent to model (if known)

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'=' * 60}")
    print(f"COORDINATE DIAGNOSTIC ANALYSIS")
    print(f"{'=' * 60}\n")

    print(f"Screen Resolution: {screen_width}x{screen_height}")
    if image_width and image_height:
        print(f"Image Resolution: {image_width}x{image_height}")

    print(f"\nRaw Model Output:\n{model_output}\n")

    # Try to extract coordinates from common formats
    import re

    # Pattern 1: [x, y] format
    pattern1 = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
    # Pattern 2: (x, y) format
    pattern2 = r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)"
    # Pattern 3: x=X, y=Y format
    pattern3 = r"x[=:]?\s*(\d+(?:\.\d+)?),?\s*y[=:]?\s*(\d+(?:\.\d+)?)"
    # Pattern 4: <box>x1,y1,x2,y2</box> format (bounding box)
    pattern4 = (
        r"<box>(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)</box>"
    )

    coords = []
    for pattern in [pattern1, pattern2, pattern3, pattern4]:
        matches = re.findall(pattern, model_output, re.IGNORECASE)
        if matches:
            coords.extend(matches)
            break

    if not coords:
        print("⚠️  WARNING: Could not extract coordinates from output!")
        print("Please manually provide coordinate values for analysis.\n")
        return {"error": "No coordinates found"}

    print(f"Extracted Coordinates: {coords}\n")

    # Analyze each coordinate pair
    for i, coord in enumerate(coords[:3]):  # Analyze up to 3 coordinates
        if len(coord) == 4:
            # Bounding box format
            x1, y1, x2, y2 = map(float, coord)
            print(f"\nBounding Box {i + 1}: [{x1}, {y1}, {x2}, {y2}]")
            analyze_coord_pair(
                x1,
                y1,
                screen_width,
                screen_height,
                image_width,
                image_height,
                label=f"Box{i + 1} Top-Left",
            )
            analyze_coord_pair(
                x2,
                y2,
                screen_width,
                screen_height,
                image_width,
                image_height,
                label=f"Box{i + 1} Bottom-Right",
            )
        else:
            # Point format
            x, y = float(coord[0]), float(coord[1])
            print(f"\nPoint {i + 1}: [{x}, {y}]")
            analyze_coord_pair(
                x, y, screen_width, screen_height, image_width, image_height
            )


def analyze_coord_pair(
    x: float,
    y: float,
    screen_width: int,
    screen_height: int,
    image_width: int = None,
    image_height: int = None,
    label: str = "Point",
):
    """Analyze a single coordinate pair to determine its format."""

    print(f"\n  {label}: ({x}, {y})")
    print(f"  " + "-" * 50)

    # Check if normalized (0.0-1.0)
    if 0 <= x <= 1 and 0 <= y <= 1:
        pixel_x = int(x * screen_width)
        pixel_y = int(y * screen_height)
        print(f"  ✓ Format: NORMALIZED (0.0-1.0)")
        print(f"  → Screen pixels: ({pixel_x}, {pixel_y})")
        if image_width and image_height:
            img_x = int(x * image_width)
            img_y = int(y * image_height)
            print(f"  → Image pixels: ({img_x}, {img_y})")

    # Check if using 0-1000 normalization (common in some models)
    elif 0 <= x <= 1000 and 0 <= y <= 1000:
        pixel_x = int((x / 1000) * screen_width)
        pixel_y = int((y / 1000) * screen_height)
        print(f"  ✓ Format: 0-1000 NORMALIZED")
        print(f"  → Screen pixels: ({pixel_x}, {pixel_y})")

    # Check if absolute pixels (matching screen resolution)
    elif x <= screen_width and y <= screen_height:
        print(f"  ✓ Format: ABSOLUTE SCREEN PIXELS")
        print(f"  → Already in screen coordinates")

    # Check if absolute pixels (matching image resolution)
    elif image_width and image_height and x <= image_width and y <= image_height:
        scale_x = screen_width / image_width
        scale_y = screen_height / image_height
        pixel_x = int(x * scale_x)
        pixel_y = int(y * scale_y)
        print(f"  ✓ Format: ABSOLUTE IMAGE PIXELS")
        print(f"  → Screen pixels: ({pixel_x}, {pixel_y})")
        print(f"  → Scale factors: X={scale_x:.3f}, Y={scale_y:.3f}")

    # Unknown format
    else:
        print(f"  ⚠️  Format: UNKNOWN or OUT OF RANGE")
        print(f"  → Coordinate exceeds expected ranges")
        if image_width and image_height:
            print(
                f"  → Exceeds screen ({screen_width}x{screen_height}) and image ({image_width}x{image_height})"
            )


def generate_conversion_code(
    from_format: str,
    to_format: str,
    screen_width: int = 1920,
    screen_height: int = 1080,
):
    """Generate Python code to convert between coordinate formats."""

    print(f"\n{'=' * 60}")
    print(f"COORDINATE CONVERSION CODE")
    print(f"{'=' * 60}\n")

    if from_format == "0-1000" and to_format == "normalized":
        print("# Convert from 0-1000 normalized to 0.0-1.0 normalized")
        print("def convert_coords(x, y):")
        print("    return x / 1000, y / 1000")

    elif from_format == "0-1000" and to_format == "pixels":
        print(f"# Convert from 0-1000 normalized to screen pixels")
        print(f"SCREEN_WIDTH = {screen_width}")
        print(f"SCREEN_HEIGHT = {screen_height}")
        print("def convert_coords(x, y):")
        print("    pixel_x = int((x / 1000) * SCREEN_WIDTH)")
        print("    pixel_y = int((y / 1000) * SCREEN_HEIGHT)")
        print("    return pixel_x, pixel_y")

    elif from_format == "normalized" and to_format == "pixels":
        print(f"# Convert from normalized (0.0-1.0) to screen pixels")
        print(f"SCREEN_WIDTH = {screen_width}")
        print(f"SCREEN_HEIGHT = {screen_height}")
        print("def convert_coords(x, y):")
        print("    pixel_x = int(x * SCREEN_WIDTH)")
        print("    pixel_y = int(y * SCREEN_HEIGHT)")
        print("    return pixel_x, pixel_y")

    elif from_format == "image_pixels" and to_format == "screen_pixels":
        print("# Convert from image pixels to screen pixels")
        print(
            "def convert_coords(x, y, image_width, image_height, screen_width, screen_height):"
        )
        print("    scale_x = screen_width / image_width")
        print("    scale_y = screen_height / image_height")
        print("    pixel_x = int(x * scale_x)")
        print("    pixel_y = int(y * scale_y)")
        print("    return pixel_x, pixel_y")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QWEN3-VL vs UI-TARS COORDINATE DIAGNOSTIC TOOL")
    print("=" * 60)

    print("\nINSTRUCTIONS:")
    print("1. Run your model on a test image with a known clickable element")
    print("2. Paste the model's coordinate output below")
    print("3. Note where the click actually lands vs where it should land")
    print(
        "4. This script will identify the coordinate format and provide conversion code"
    )

    print("\n" + "=" * 60)
    print("EXAMPLE USAGE:")
    print("=" * 60)

    # Example 1: Qwen3-VL output
    print("\n### Example 1: Qwen3-VL Output ###")
    example_output_qwen3 = "Click at coordinates [523, 678] to open the browser"
    analyze_coordinate_output(
        model_output=example_output_qwen3,
        screen_width=1920,
        screen_height=1080,
        image_width=1920,
        image_height=1080,
    )

    # Example 2: UI-TARS output (normalized)
    print("\n\n### Example 2: UI-TARS Output ###")
    example_output_uitars = "Click at coordinates [0.272, 0.628] to open the browser"
    analyze_coordinate_output(
        model_output=example_output_uitars, screen_width=1920, screen_height=1080
    )

    # Generate conversion code
    generate_conversion_code(from_format="0-1000", to_format="pixels")

    print("\n\n" + "=" * 60)
    print("TO USE WITH YOUR MODELS:")
    print("=" * 60)
    print("\n1. Modify the example outputs with your actual model outputs")
    print("2. Update screen_width and screen_height with your display resolution")
    print("3. Run this script to identify the coordinate format mismatch")
    print("4. Use the generated conversion code in your UI-TARS desktop integration")
    print("\n")
