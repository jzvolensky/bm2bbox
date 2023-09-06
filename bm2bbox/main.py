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
                        type=str,
                        default=(0,255,0),
                        metavar='',
                        help='string. e.g. "0,255,0" in BGR format. Color of the object in the binary mask'
                        )
    parser.add_argument('-background_color',
                        type=str,
                        default=(255,0,255),
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
    parser.add_argument('-debug',
                        type=bool,
                        default=False,
                        metavar='',
                        help='bool. Debug mode'
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

    return corrected_image

def draw_bbox(corrected_image, object_color, background_color):
    '''
    Function to draw a bounding box around a binary mask or a set of binary masks
    '''

    height, width, _ = corrected_image.shape
    blank_image = np.zeros((height, width, 4), dtype=np.uint8) 
    blank_image[..., 3] = 0  

    hsv_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)

    object_color_values = object_color.split(',')
    if len(object_color_values) == 3:
        object_rgb = tuple(map(int, object_color_values))
    else:
        print("Invalid object_color format. Expected format: '85,232,249'")


    background_color_values = background_color.split(',')
    if len(background_color_values) == 3:
        background_rgb = tuple(map(int, background_color_values))
    else:
        print("Invalid background_color format. Expected format: '85,232,249'")

    object_hsv = cv2.cvtColor(np.uint8([[object_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    color_margin_percentage = 10

    hue_margin = int((object_hsv[0] / 360) * color_margin_percentage)
    saturation_margin = int((object_hsv[1] / 255) * color_margin_percentage)
    value_margin = int((object_hsv[2] / 255) * color_margin_percentage)

    lower_color = np.array([
        max(0, object_hsv[0] - hue_margin),
        max(0, object_hsv[1] - saturation_margin),
        max(0, object_hsv[2] - value_margin)
    ], dtype=np.uint8)

    upper_color = np.array([
        min(179, object_hsv[0] + hue_margin),
        min(255, object_hsv[1] + saturation_margin),
        min(255, object_hsv[2] + value_margin)
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    object_color_values = object_color.split(',')
    object_rgb = tuple(map(int, object_color_values))
    object_b, object_g, object_r = object_rgb
    print(f'Object Color (BGR): ({object_b}, {object_g}, {object_r})')

    background_color_values = background_color.split(',')
    background_rgb = tuple(map(int, background_color_values))
    background_b, background_g, background_r = background_rgb
    print(f'Background Color (BGR): ({background_b}, {background_g}, {background_r})')

    return blank_image



def save_geojson(output_path, bounding_boxes):
    with open(output_path, "w") as geojson_file:
        json.dump(bounding_boxes, geojson_file, indent=2)


def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output
    output_folder = args.output_folder
    object_color = args.object_color
    background_color = args.background_color
    single_image = args.s
    debug_mode = args.debug

    if single_image:
        image = prepare_single_image(input_path)
        bounding_box = draw_bbox(image, object_color, background_color)
        
        # Specify the output file format based on the provided output_path
        if output_path is None:
            output_path = "output.png"  # Default output file name
        
        # Ensure the output file path has a valid extension
        if not output_path.endswith((".png", ".jpg", ".jpeg")):
            print("Invalid output file format. Supported formats: .png, .jpg, .jpeg")
            return

        cv2.imwrite(output_path, bounding_box)
    else:
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = prepare_images_folder(input_path)
        for i, image in enumerate(images):
            bounding_box = draw_bbox(image, object_color, background_color)
            
            # Specify the output file format based on the image format
            output_file = os.path.join(output_folder, f"output_{i}.png")
            
            cv2.imwrite(output_file, bounding_box)

    if debug_mode == True:
        print("Debug mode is on")
        print("Input path: ", input_path)
        print("Output path: ", output_path)
        print("Output folder: ", output_folder)
        print("Object color: ", object_color)
        print("Background color: ", background_color)
        print("Single image?: ", single_image)


if __name__ == "__main__":
    main()

