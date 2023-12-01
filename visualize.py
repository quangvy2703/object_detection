from rml.utils.visualization import draw_coco_bboxes

image_path = "/Users/phamvy/Projects/dataset/furniture/lvis/train/images/000000000257.jpg"
annotation_path = "/Users/phamvy/Projects/dataset/furniture/lvis/train/labels/000000000257.txt"

draw_coco_bboxes(image_path, annotation_path)