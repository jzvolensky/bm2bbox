import os
import cv2
import pytest
import geojson
from bm2bbox.main import draw_bbox, prepare_images_folder, prepare_single_image

current_dir = os.path.dirname(os.path.abspath(__file__))

image_folder = os.path.join(current_dir, "test_data", "image_folder")

image_file_names = [
    "test_image1.png",
    "test_image2.png",
    "test_image3.png",
    "test_image4.png",
    "test_image5.jpeg",
]

test_image_paths = [os.path.join(image_folder, image_file) for image_file in image_file_names]


@pytest.mark.parametrize("test_image_path", test_image_paths)
def test_draw_bbox_output(test_image_path):

    image = cv2.imread(test_image_path)

    geojson_features = draw_bbox(image, val=50, debug_mode=False)

    assert geojson_features  

    output_dir = "tests/test_results"

    output_geojson_path = os.path.join(output_dir, "test_output.geojson")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_geojson_path, "w") as geojson_file:
        geojson.dump(geojson.FeatureCollection(geojson_features), geojson_file, indent=2)

    with open(output_geojson_path, "r") as geojson_file:
        for i, line in enumerate(geojson_file):
            if i < 5:  
                print(line.strip())


test_dir = os.path.dirname(os.path.abspath(__file__))
single_image_path = os.path.join(test_dir, "test_data/single_image/test_bm.png")
folder_path = os.path.join(test_dir, "test_data/image_folder")

@pytest.fixture
def single_image():
    # Prepare a single image for testing
    image, image_path = prepare_single_image(single_image_path, debug_mode=False)
    return image

@pytest.fixture
def folder_images():
    # Prepare images from a folder for testing
    images, image_paths = prepare_images_folder(folder_path)
    return images

def test_single_image_processing(single_image):
    # Test processing a single image
    geojson_features = draw_bbox(single_image, val=50, debug_mode=False)

    # Assert that the generated GeoJSON is not empty
    assert geojson_features

def test_folder_image_processing(folder_images):
    # Test processing a folder of images
    for i, image in enumerate(folder_images):
        geojson_features = draw_bbox(image, val=50, debug_mode=False)

        # Assert that the generated GeoJSON is not empty
        assert geojson_features
