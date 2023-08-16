from PIL import Image
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
    x2, y2 = x1 + overlay_img.width, y1 + overlay_img.height

    # Alpha blending
    alpha_base = base_np[y1:y2, x1:x2, 3] / 255.0
    alpha_overlay = overlay_np[:, :, 3] / 255.0
    alpha_out = alpha_overlay + alpha_base * (1 - alpha_overlay)

    base_np[y1:y2, x1:x2, 3] = alpha_out * 255
    base_np[y1:y2, x1:x2, :3] = (
        (overlay_np[:, :, :3] * alpha_overlay[:, :, np.newaxis]) +
        (base_np[y1:y2, x1:x2, :3] * alpha_base[:, :, np.newaxis] * (1 - alpha_overlay[:, :, np.newaxis]))
    )

    return Image.fromarray(base_np)

def build_piece(piece, data):
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
            # Parameter
            if image_name[0] == '*':
                param = image_name[1:]
                image_name = piece["parameters"][param]

            # Check for empty image
            if len(image_name) == 0:
                continue

            # Open image
            image = Image.open(f"images/{image_name}.{extension}").convert('RGBA')

            # Layer the image over the combined image
            combined_image = blend_images(combined_image, image, (0, 0))

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
        columns_per_page = int((page_width + piece_margin) / (piece_width + piece_margin))
        rows_per_page = int((page_height + piece_margin) / (piece_height + piece_margin))
        pieces_per_page = columns_per_page * rows_per_page
        pages = math.ceil(len(piece_images["front"]) / pieces_per_page)

        # Arrange images on page
        for i in range(pages):
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

                paste_x = (x * piece_width) + (x * piece_margin)
                paste_y = (y * piece_height) + (y * piece_margin)

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


main()
