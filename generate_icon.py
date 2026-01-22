from PIL import Image, ImageDraw
import os

def create_icon(input_path, output_path, size=(256, 256), radius=40, padding=20):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    try:
        img = Image.open(input_path)
        img = img.convert("RGBA")
        
        # Calculate aspect ratio
        width, height = img.size
        aspect = width / height
        
        # Effective size after padding
        target_w = size[0] - (padding * 2)
        target_h = size[1] - (padding * 2)
        
        # New dimensions fitting within target size
        if aspect > 1:
            new_w = target_w
            new_h = int(new_w / aspect)
        else:
            new_h = target_h
            new_w = int(new_h * aspect)
            
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create transparent canvas for the image
        img_canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        
        # Center image with padding offset
        x = (size[0] - new_w) // 2
        y = (size[1] - new_h) // 2
        
        img_canvas.paste(img_resized, (x, y), img_resized)
        
        # Create Rounded Mask on the FULL size
        # We want the mask to cut the image, but maybe we want the shape itself to be smaller?
        # If we just mask the padded image, the result is a small image floating in space.
        # But if the User wants "Rounded Corners" on the ICON, they usually mean the shape of the icon.
        # Let's apply the mask to the PADDED area.
        
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw rounded rect slightly smaller than full size if we want visual padding from desktop grid
        # But simpler: Draw rounded rect matching the image size or the full box?
        # Let's draw the mask matching the full box, but because we padded the image, 
        # the image is effectively smaller.
        # Wait, if we pad the image, the corners of the *image* are what matters? 
        # Usually icons are full-bleed. 
        # Let's interpret "Size too big" as "It touches the edges too much".
        # So we keep the mask full (or slightly reduced) but shrink the image content?
        # Actually simplest: Reduce the mask size too.
        
        rect_x0 = padding
        rect_y0 = padding
        rect_x1 = size[0] - padding
        rect_y1 = size[1] - padding
        
        draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], radius=radius, fill=255)
        
        # Apply mask
        final_icon = Image.new("RGBA", size, (0, 0, 0, 0))
        final_icon.paste(img_canvas, (0, 0), mask=mask)
        
        final_icon.save(output_path)
        print(f"Successfully created padded rounded {output_path}")
        
    except Exception as e:
        print(f"Error creating icon: {e}")

if __name__ == "__main__":
    create_icon("logo.png", "icon.png")
