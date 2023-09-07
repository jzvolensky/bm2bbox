# bm2bbox
A simple tool to generate bounding boxes and coordinates for machine learning


# Examples

*Image Folder with multiple images*

bm2bbox -input "path/to/image_folder" -output "path/to/output/folder" 

```bash
bm2bbox -input "/home/jzvolensky/Documents/bm2bbox/test_data/image_folder" -output "/home/jzvolensky/Documents/bm2bbox/test_output"
```

*Single image*

bm2bbox -input "path/to/image.png" -output "path/to/output/output.json

```bash
bm2bbox -input "/home/jzvolensky/Documents/bm2bbox/test_data/single_image/test_bm.png" -s -output "/home/jzvolensky/Documents/bm2bbox/test_output"
```
