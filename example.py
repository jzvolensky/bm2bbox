import bbox_tool
from bbox_tool.data_processing import data_processor
from bbox_tool.data_processing.data_processor import load_and_prep, draw_bounding_boxes
from bbox_tool.data_processing.data_processor import draw_bounding_boxes

from bbox_tool.save_output import save_output

# --object_color "fde724" --background_color "440458"

folder_path = '/Users/jzvolensky/ml-bbox-tool/test_data'
object_color = 'fde724'
background_color = '440458'
box_color = (0, 255, 0)

preprocessed_images = load_and_prep(folder_path)

processed_images = draw_bounding_boxes(preprocessed_images, object_color, background_color, box_color)