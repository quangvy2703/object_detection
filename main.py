import argparse
import os
import json

from rml.model_loader.yolo_model_loader import YOLOv8ModelLoader
from rml.domain.inference_input import ObjectDetectionInferenceInput


def main(args):
    model_loader = YOLOv8ModelLoader.from_pretrained(
        model_path='rml/data/models/yolov8n.pt'
    )
    train_configs = YOLOv8ModelLoader.load_training_config(args.train_config_path)
    args.training_data_config_paths = [item.strip() for item in args.training_data_config_paths.split(',')]
    args.data_dirs = [item.strip() for item in args.data_dirs.split(',')]

    YOLOv8ModelLoader.update_data_config_file(
        data_config_files=args.training_data_config_paths,
        paths=args.data_dirs
    )

    train_configs = YOLOv8ModelLoader.merge_configs(train_configs, vars(args))

    print("Dir info")
    if os.path.exists("/opt/ml"):
        print(f"Debug info: /opt/ml ", os.listdir("/opt/ml"))
    if os.path.exists("/opt/ml/datasets"):
        print(f"Debug info: /opt/ml/datasets ", os.listdir("/opt/ml/datasets"))
    if os.path.exists("/opt/ml/datasets/rever"):
        print(f"Debug info: /opt/ml/datasets/rever ", os.listdir("/opt/ml/datasets/rever"))

    model_loader.train(
        training_data_config_paths=args.training_data_config_paths,
        train_configs=train_configs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        metavar="N",
        help="using pretrained model",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=10,
        metavar="N",
        help="",
    )

    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    parser.add_argument(
        "--training_data_config_paths",
        type=str,
        metavar="N",
        help="training datasets configs",
    )

    parser.add_argument(
        "--train_config_path",
        type=str,
        metavar="N",
        help="training configs",
    )

    parser.add_argument(
        "--data_dirs",
        type=str,
        metavar="N",
        help="data directory",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        metavar="N",
        help="output directory",
    )

    main(parser.parse_args())

    # detected = model_loader.inference(
    #     inference_input=ObjectDetectionInferenceInput.from_urls([
    #         "https://photo.rever.vn/v3/get/rvk1YDxeNaQRcrIc_tsCaOeGFqwI1vrf5J7CPEnbZdIGTDnNnA7CAAG+iW3AJuMbBaE76MkTuDuQQF7M35VYC0Iw==/900x600/image.jpg",
    #         "https://photo.rever.vn/v3/get/rv3oj3knx5+QDK2tbxH+kshZeinoeWg6p1ugc5i9zyKon+6rpXp3Q6sP6u7+0VJHm+fo6vCH2PtzUdVMuIVeTq5g==/900x600/image.jpg"
    #     ]),
    #     save=True
    # )
