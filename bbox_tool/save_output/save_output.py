import os
import cv2


def save_images(processed_images, input_folder):
    output_folder = os.path.join(input_folder, 'output_images')
    os.makedirs(output_folder, exist_ok=True)

    for index, image in enumerate(processed_images):
        output_path = os.path.join(output_folder, f'processed_{index}.png')
        cv2.imwrite(output_path, image)

    return output_folder