import argparse
import os
import cv2
import json
import numpy as np

def parse_args():
    '''
    Argparse function to parse the input arguments
    cli parameters: 
    --input: path to input binary mask
    --s: use if input is a single image
    --object_color: color of the object in the binary mask
    --background_color: color of the background in the binary mask
    --output: path to output bounding box
    --output_folder: path to output folder
    '''
    parser = argparse.ArgumentParser(description='Converts a binary mask to a bounding box')
    parser.add_argument('-input',
                        type=str,
                        metavar='',
                        help='string. Path to input binary mask'
                        )
    parser.add_argument('-s',
                        type=bool,
                        default=True,
                        metavar='',
                        help='bool. Use if input is a single image'
                        )

    parser.add_argument('-object_color',
                        type=int,
                        default=(0,255,0),
                        metavar='',
                        help='string. e.g. "0,255,0" in BGR format. Color of the object in the binary mask'
                        )
    parser.add_argument('-background_color',
                        type=int,default=(255,0,255),
                        metavar='',
                        help='string. e.g. "255,0,255" in BGR format. Color of the background in the binary mask'
                        )

    parser.add_argument('-output',
                        type=str,
                        metavar='',
                        help='string. Path to output bounding box'
                        )
    parser.add_argument('-output_folder',
                        type=str,
                        metavar='',
                        help='string. Path to output bounding box folder'
                        )

    return parser.parse_args()

def prepare_single_image(image_path):
    '''
    Loads an image from a path and converts it 
    to RGBA if it is not already in RGBA format
    '''
    image = cv2.imread(image_path)
    corrected_image = []
    if image.shape[2] == 3:
        corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] == 4:
        corrected_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    corrected_image.append(image)

    return corrected_image

def prepare_images_folder(folder_path):
    '''
    Loads images from a folder and converts them 
    to RGBA if they are not already in RGBA format
    '''
    image_files = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, img_file) for img_file in image_files if img_file.endswith('.png')]
    corrected_image = []
    for image_path in image_paths:
        image = cv2.imread(image_path)

        if image.shape[2] == 3:
            corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            corrected_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        corrected_image.append(image)

    return corrected_image

def draw_bbox(corrected_image, object_color, background_color, box_color):
    '''
    Function to draw a bounding box around a binary mask or a set of binary masks
    '''



    blank_image = np.ones(corrected_image.shape, dtype=np.uint8) * 255
    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2BGRA)
    blank_image[..., 3] = 0  # Set the alpha channel to 0 for transparency

    hsv_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)

    object_rgb = tuple(int(object_color[i:i+2], 16) for i in (0, 2, 4))
    background_rgb = tuple(int(background_color[i:i+2], 16) for i in (0, 2, 4))
    
    object_hsv = cv2.cvtColor(np.uint8([[object_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    lower_color = np.array([object_hsv[0]-10, 50, 50], dtype=np.uint8)
    upper_color = np.array([object_hsv[0]+10, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(blank_image, (x, y), (x + w, y + h), box_color, 2)
    
    return blank_image

def save_geojson(output_path, bounding_boxes):
    with open(output_path, "w") as geojson_file:
        json.dump(bounding_boxes, geojson_file, indent=2)

if __name__ == "__main__":
    args = parse_args()

    input_path = args.input
    output_path = args.output
    output_folder = args.output_folder
    object_color = args.object_color
    background_color = args.background_color
    single_image = args.s

def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output
    output_folder = args.output_folder
    object_color = args.object_color
    background_color = args.background_color
    single_image = args.s

    if single_image==True:
        image = prepare_single_image(input_path)
        bounding_box = draw_bbox(image, object_color, background_color)
        cv2.imwrite(output_path, bounding_box)
    else:
        images = prepare_images_folder(input_path)
        for image in images:
            bounding_box = draw_bbox(image, object_color, background_color)
            cv2.imwrite(output_folder, bounding_box)