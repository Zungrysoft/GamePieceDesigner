from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import json
import math

def blend_images(base_img, overlay_img, position=(0, 0)):
    base_img = base_img.convert("RGBA")
    overlay_img = overlay_img.convert("RGBA")

    # Convert PIL Images to NumPy arrays and explicitly copy them
    base_np = np.array(base_img).copy()
    overlay_np = np.array(overlay_img).copy()

    x1, y1 = position
    x2, y2 = min(x1 + overlay_img.width, base_img.width), min(y1 + overlay_img.height, base_img.height)

    # Determine overlay slicing based on position (handles negative offsets)
    ox1, oy1 = max(0, -x1), max(0, -y1)
    ox2, oy2 = overlay_img.width + min(0, base_img.width - x1 - overlay_img.width), overlay_img.height + min(0, base_img.height - y1 - overlay_img.height)

    # Adjust base image slicing to remain within valid bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, base_img.width), min(y2, base_img.height)

    # Alpha blending
    alpha_base = base_np[y1:y2, x1:x2, 3] / 255.0
    alpha_overlay = overlay_np[oy1:oy2, ox1:ox2, 3] / 255.0
    alpha_out = alpha_overlay + alpha_base * (1 - alpha_overlay)

    base_np[y1:y2, x1:x2, 3] = (alpha_out * 255).astype(np.uint8)
    base_np[y1:y2, x1:x2, :3] = (
        (overlay_np[oy1:oy2, ox1:ox2, :3] * alpha_overlay[:, :, np.newaxis]) +
        (base_np[y1:y2, x1:x2, :3] * alpha_base[:, :, np.newaxis] * (1 - alpha_overlay[:, :, np.newaxis]))
    ).astype(np.uint8)

    return Image.fromarray(base_np)

def convert_value(value, parameters):
    # Convert from value to parameter if starts with *
    if value[0] == '*':
        param = value[1:]
        return parameters[param] if param in parameters else ""

    # Normal value
    return value

def wrap_text(text, font_size, width):
    width_factor = 0.5

    wrapped_text = ""
    word = ""
    line_chars = 0
    lines = 1
    text += "\n"
    for char in text:
        if char in [' ', '-', '\n']:
            if line_chars * font_size * width_factor >= width:
                wrapped_text += '\n' + word + char
                line_chars = len(word) + 1
                word = ""
                lines += 1
            else:
                wrapped_text += word + char
                word = ""
                line_chars += 1
        else:
            word += char
            line_chars += 1

    return wrapped_text, lines

def fit_text(text, max_font_size, x1, y1, x2, y2):
    height_factor = 1.167

    # Calculate box width and height
    width = abs(x1 - x2)
    height = abs(y1 - y2)

    # Keep trying smaller and smaller font sizes until the text fits
    font_size = max_font_size
    while font_size > 1:
        wrapped_text, lines = wrap_text(text, font_size, width)

        # Check if the total height of the text exceeds the box height
        if lines * font_size * height_factor <= height:
            return wrapped_text, font_size
        else:
            font_size -= 1

    # Return 1 as a fallback
    return 1

def build_piece(piece, data):
    # Print
    if "parameters" in piece and "name" in piece["parameters"]:
        print(f'Generating [{piece["parameters"]["name"]}]...')

    # Get prototype for this piece
    prototype = data["prototypes"][piece["prototype"]]

    # Get image extension
    extension = data["settings"]["image_extension"]

    # Iterate over outputs in prototype
    results = {
        "front": [],
        "back": []
    }
    for side in ["front", "back"]:
        if not side in prototype:
            if side == "back" and "copy_front" in prototype and prototype["copy_front"]:
                results["back"].append(results["front"][-1])
            else:
                results[side].append(None)
            continue
        output = prototype[side]

        # Initialize image
        combined_image = Image.new('RGBA', (data["settings"]["piece_width_px"], data["settings"]["piece_height_px"]))

        # Add image layers
        for image_name in output["images"]:
            # Convert if it is a special value
            image_name = convert_value(image_name, piece["parameters"])

            # Check for empty image
            if len(image_name) == 0:
                continue

            # Open image
            image = Image.open(f"images/{image_name}.{extension}").convert('RGBA')

            # Get image offset
            x = 0
            y = 0
            if "image_settings" in data and image_name in data["image_settings"]:
                if "x" in data["image_settings"][image_name]:
                    x = data["image_settings"][image_name]["x"]
                if "y" in data["image_settings"][image_name]:
                    y = data["image_settings"][image_name]["y"]

            # Layer the image over the combined image
            combined_image = blend_images(combined_image, image, (x, y))

        # Add text layers
        for box_settings in output["text_boxes"]:
            # Get text box data
            box = data["text_boxes"][box_settings["type"]]

            # Set parameters
            font_size = int(box["font_size"]) if "font_size" in box else 30
            color = box["color"] if "color" in box else "black"
            x = box["x"] if "x" in box else 0
            y = box["y"] if "y" in box else 0
            x2 = box["x2"] if "x2" in box else 512
            y2 = box["y2"] if "y2" in box else 512
            text = str(convert_value(box_settings["text"], piece["parameters"])) if "text" in box_settings else ""

            # Text anchoring
            anchor_map = {
                "left": "lm",
                "right": "rm",
                "center": "mm",
                "box": None
            }
            anchor = anchor_map[box["mode"]]

            # Text wrapping for box mode
            if box["mode"] == "box":
                text, font_size = fit_text(text, font_size, x, y, x2, y2)

            font = ImageFont.truetype("Tests/fonts/NotoSans-Regular.ttf", font_size)
            d = ImageDraw.Draw(combined_image)
            d.text((x, y), text, fill=color, anchor=anchor, font=font)

        # Append to results
        results[side].append(combined_image)

    # Multiply output from count
    if "count" in piece:
        results["front"] *= piece["count"]
        results["back"] *= piece["count"]
    if "count" in prototype:
        results["front"] *= prototype["count"]
        results["back"] *= prototype["count"]

    # Return
    return results

def main():
    # Check args
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [settings.json]")
        return

    # Open JSON file
    json_file = sys.argv[1]
    with open(json_file, 'r') as input_file:
        data = json.load(input_file)

        # Build each piece
        piece_images = {
            "front": [],
            "back": []
        }
        for piece in data["pieces"]:
            results = build_piece(piece, data)
            piece_images["front"] += results["front"]
            piece_images["back"] += results["back"]

        # Determine page layout parameters
        pixels_per_cm = data["settings"]["piece_width_px"] / data["settings"]["piece_width_cm"]
        piece_width = int(data["settings"]["piece_width_px"])
        piece_height = int(data["settings"]["piece_height_px"])
        page_width = int(data["settings"]["page_width_cm"] * pixels_per_cm)
        page_height = int(data["settings"]["page_height_cm"] * pixels_per_cm)
        piece_margin = int(data["settings"]["piece_margin_cm"] * pixels_per_cm)
        edge_margin = int(data["settings"]["edge_margin_cm"] * pixels_per_cm)
        columns_per_page = int(((page_width - (edge_margin*2)) + piece_margin) / (piece_width + piece_margin))
        rows_per_page = int(((page_height - (edge_margin*2)) + piece_margin) / (piece_height + piece_margin))
        pieces_per_page = columns_per_page * rows_per_page
        pages = math.ceil(len(piece_images["front"]) / pieces_per_page)

        # Arrange images on page
        for i in range(pages):
            print(f"Arranging page {i+1}...")
            # Create page image
            page_image_front = Image.new('RGBA', (page_width, page_height))
            page_image_back = Image.new('RGBA', (page_width, page_height))

            # Track edits
            page_edits_front = 0
            page_edits_back = 0

            # Iterate over piece images
            for j in range(pieces_per_page):
                # Get piece image
                piece_index = (i * pieces_per_page) + j
                if piece_index >= len(piece_images["front"]):
                    break
                piece_image_front = piece_images["front"][piece_index]
                piece_image_back = piece_images["back"][piece_index]

                # Align image
                x = j % columns_per_page
                y = int(j / columns_per_page)

                paste_x = (x * piece_width) + (x * piece_margin) + edge_margin
                paste_y = (y * piece_height) + (y * piece_margin) + edge_margin

                if piece_image_front:
                    page_image_front.paste(piece_image_front, (paste_x, paste_y), mask=piece_image_front)
                    page_edits_front += 1
                if piece_image_back:
                    page_image_back.paste(piece_image_back, (page_width - paste_x - piece_width, paste_y), mask=piece_image_back)
                    page_edits_back += 1

            # Output page image
            extension = data["settings"]["image_extension"]
            if page_edits_front > 0:
                page_image_front.save(f"results/page_{i+1}_front.{extension}")
            if page_edits_back > 0:
                page_image_back.save(f"results/page_{i+1}_back.{extension}")

        print("Done!")


main()
