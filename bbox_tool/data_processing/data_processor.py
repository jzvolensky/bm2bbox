from bbox_tool import data_input
import numpy as np
import cv2
import os
import bbox_tool


def load_and_prep(folder_path):
    binary_masks = bbox_tool.LoadFolder(folder_path)
    preprocessed_images = []
    for image_path in binary_masks:
        image = cv2.imread(image_path)

        # Convert image to RGBA if needed
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        preprocessed_images.append(image)
    return preprocessed_images


def draw_bounding_boxes(folder_path, object_color, background_color, box_color, output_image=None):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                   filename.lower().endswith('.png')]

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)

        # Create a blank image with a white background and transparent alpha channel
        blank_image = np.ones(image.shape, dtype=np.uint8) * 255
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2BGRA)
        blank_image[..., 3] = 0  # Set the alpha channel to 0 for transparency

        # Convert the image to HSV color space for easier color detection
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Convert the object and background colors to RGB
        object_rgb = tuple(int(object_color[i:i + 2], 16) for i in (0, 2, 4))
        background_rgb = tuple(int(background_color[i:i + 2], 16) for i in (0, 2, 4))

        # Convert the object and background colors to HSV
        object_hsv = cv2.cvtColor(np.uint8([[object_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        background_hsv = cv2.cvtColor(np.uint8([[background_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        # Define the lower and upper color thresholds for the object color
        lower_color = np.array([object_hsv[0] - 10, 50, 50], dtype=np.uint8)
        upper_color = np.array([object_hsv[0] + 10, 255, 255], dtype=np.uint8)

        # Threshold the image to extract pixels within the specified color range
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        # Find contours of the objects within the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes outline on the blank image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(blank_image, (x, y), (x + w, y + h), box_color, 2)

        if output_image is None:
            image_filename = os.path.basename(image_path)
            default_output_image = os.path.join(os.path.dirname(image_path), "output", image_filename)
            default_output_image = os.path.splitext(default_output_image)[0] + "_output.png"
            output_image = default_output_image

        cv2.imwrite(output_image, blank_image)
