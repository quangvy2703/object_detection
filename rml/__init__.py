"""
rever-sagemaker
    furniture_detection
        best_model
            training-info.json
            model.pt
            metrics.jsons
            metrics.jpeg
        best_model
            training-info.json
            model.pt
            metrics.jsons
            metrics.jpeg

"""
"""
python main_detection.py \
--train_config_path="rml/configs/object_detection/default_training_config.yaml" \
--training_data_config_paths="rml/configs/object_detection/open_image_v7_furniture.yaml, rml/configs/object_detection/rever_furniture.yaml" \
--data_dirs="/Users/phamvy/Projects/dataset/furniture_dataset, /Users/phamvy/Desktop/rever" \
--pretrained_path="rml/data/models/yolov8n.pt" \
--epochs=50
"""

"""
python main_detection.py \
--train_config_path="rml/configs/object_detection/default_training_config.yaml" \
--training_data_config_paths=" rml/configs/object_detection/rever_furniture.yaml" \
--data_dirs="/Users/phamvy/Desktop/rever" \
--pretrained_path="rml/data/models/yolov8n.pt" \
--epochs=50
"""
