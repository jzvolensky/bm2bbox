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
    if image is None:
        print('Could not open or find the image:', args.input)
        exit(0)
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

#TODO: Remove the temporary return statement and integrate the saving process into the whole app
#TODO: Test on different colored images to see if the object and background colors are correctly detected
#TODO: If yes remove the object_color and background_color arguments

def draw_bbox(corrected_image, val):
    '''
    Function to draw a bounding box around a binary mask or a set of binary masks
    '''
    src_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    threshold = val

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (0, 255, 0)
        #cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 1)
        #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv2.imwrite('output.png', drawing)

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
        bounding_box = draw_bbox(image,val=100)
        
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

