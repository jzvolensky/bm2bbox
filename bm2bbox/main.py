import argparse
import os
import cv2
import json
import numpy as np
import geojson

def parse_args():
    '''
    Argparse function to parse the input arguments
    cli parameters: 
    --input: path to input binary mask
    --s: use if input is a single image
    --output: path to output bounding box
    '''
    parser = argparse.ArgumentParser(description='Converts a binary mask to a bounding box')
    parser.add_argument('-input',
                        type=str,
                        metavar='',
                        help='string. Path to input binary mask'
                        )
    parser.add_argument('-s',
                        action='store_true',
                        default=False,
                        help='bool. Use if input is a single image'
                        )
    parser.add_argument('-output',
                        type=str,
                        metavar='',
                        help='string. Path to output bounding box'
                        )

    parser.add_argument('-debug',
                        action='store_true',
                        default=False,
                        help='bool. Debug mode'
                        )

    return parser.parse_args()

def prepare_single_image(image_path,debug_mode=False):
    '''
    Loads an image from a path and converts it 
    to RGBA if it is not already in RGBA format
    '''
    image = cv2.imread(image_path)
    if image is None:
        print("Invalid image path")
        exit(0)
    corrected_image = []
    if image.shape[2] == 3:
        corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] == 4:
        corrected_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    if debug_mode:
        print("Debug mode is on")
        print("prepare_single_image function is running")
        print("corrected_image datatype: ", corrected_image.dtype)
        print("corrected_image shape: ", corrected_image.shape)
        print("corrected_image size: ", corrected_image.size)
        print("corrected_image ndim: ", corrected_image.ndim)
        print("prepare_single_image function is done")
        print("====================================")

    return corrected_image, image_path

def prepare_images_folder(folder_path):
    '''
    Loads images from a folder and converts them 
    to RGBA if they are not already in RGBA format
    '''
    image_files = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, img_file) for img_file in image_files if img_file.endswith('.png')]
    corrected_images = []  

    for image_path in image_paths:
        image = cv2.imread(image_path)

        if image.shape[2] == 3:
            corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            corrected_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        corrected_images.append(corrected_image) 

    return corrected_images, image_paths

def draw_bbox(corrected_image, val, debug_mode=False):
    src_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    threshold = val

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    geojson_features = []
    bbox_id = 0

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        feature = geojson.Feature(
            geometry=geojson.Polygon([[
                (x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)
            ]]),
            properties={
                "id": bbox_id,
            }
        )

        geojson_features.append(feature)
        bbox_id += 1

    if debug_mode:
        print("draw_bbox function is running")
        print("draw_bbox function is done")
        print("====================================")


    return geojson_features
        


def export_geojson(drawing, output_path):
    '''
    Function to export the bounding box as a geojson annotation file.
    '''
    with open(output_path, "w") as geojson_file:
        json.dump(drawing, geojson_file, indent=2)

def main():
    args = parse_args()

    input_path = args.input
    single_image = args.s
    debug_mode = args.debug

    if single_image:

        image, image_path = prepare_single_image(input_path, debug_mode=debug_mode)
        images = [image]


        input_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_file = f'bbox_{input_filename}.geojson'


        output_path = args.output or os.path.join("output", output_file)

        geojson_features = draw_bbox(image, val=50, debug_mode=debug_mode)

        if os.path.isdir(output_path):
     
            output_path = os.path.join(output_path, output_file)

 
        with open(output_path, 'w') as geojson_file:
            geojson.dump(geojson.FeatureCollection(geojson_features), geojson_file, indent=2)

        if debug_mode:
            print(f"Saved GeoJSON: {output_path}")

    else:
     
        images, image_paths = prepare_images_folder(input_path)


        output_folder = args.output or "output"

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for i, image in enumerate(images):
            output_path = None  

            geojson_features = draw_bbox(image, val=50, debug_mode=args.debug)

          
            input_filename = os.path.splitext(os.path.basename(image_paths[i]))[0]
            output_path = os.path.join(output_folder, f'bbox_{input_filename}.geojson')

         
            if os.path.isdir(output_path):
        
                output_path = os.path.join(output_path, f'bbox_{input_filename}.geojson')

            with open(output_path, 'w') as geojson_file:
                geojson.dump(geojson.FeatureCollection(geojson_features), geojson_file, indent=2)

            if debug_mode:
                print(f"Saved GeoJSON: {output_path}")

if __name__ == "__main__":
    main()
