import argparse
import os
import json

from rml.model_loader.yolov8 import YOLOv8ModelLoader
from rml.utils.on_train_end import OnTrainEnd
from rml.domain.inference_input import InferenceInput


def main(args):
    print("main", args)
    model_loader = YOLOv8ModelLoader.from_pretrained(
        model_path=args.pretrained_path,
        task=YOLOv8ModelLoader.DETECTION
    )
    delimiter = ","
    args.training_data_config_paths = [item.strip() for item in args.training_data_config_paths.split(delimiter)]
    args.data_dirs = [item.strip() for item in args.data_dirs.split(delimiter)]
    args.metrics = [item.strip() for item in args.metrics.split(delimiter)]
    args.device = [int(item.strip()) for item in args.device.split(delimiter)] if args.device else "cpu"
    YOLOv8ModelLoader.update_data_config_file(
        data_config_files=args.training_data_config_paths,
        data_dirs=args.data_dirs
    )

    train_configs = YOLOv8ModelLoader.load_training_config(args.train_config_path)
    train_configs = YOLOv8ModelLoader.merge_configs(train_configs, vars(args))
    print(train_configs)
    model_loader.train(
        training_data_config_paths=args.training_data_config_paths,
        train_configs=train_configs
    )

    # if hasattr(args, "remote_save_dir"):
    #     OnTrainEnd(
    #         local_saved_dir=train_configs["project"],
    #         remote_saved_dir=args.remote_save_dir
    #     ).on_train_end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        # metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        # metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="yolov8l-oiv7.pt",
        # metavar="N",
        help="pretrained model path",
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        # metavar="N",
        help="using pretrained model",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="furniture-detection",
        # metavar="N",
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
        default="rml/configs/object_detection/rever_furniture.yaml, rml/configs/object_detection/open_image_v7_furniture.yaml, rml/configs/object_detection/lvis_furniture.yaml",
        # metavar="N",
        help="training datasets configs",
    )

    parser.add_argument(
        "--train_config_path",
        type=str,
        default="rml/configs/object_detection/default_training_config.yaml",
        # metavar="N",
        help="training configs",
    )

    parser.add_argument(
        "--data_dirs",
        type=str,
        default="/kaggle/input/furniture/furniture/furniture/rever, /kaggle/input/furniture/furniture/furniture/open_images, /kaggle/input/furniture/lvis/lvis",
        # metavar="N",
        help="data directory",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="furniture_model_2",
        # metavar="N",
        help="output directory",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="precision, recall, average_precision",
        # metavar="N",
        help="using metrics",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0, 1",
        # metavar="N",
        help="using metrics",
    )

    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        # metavar="N",
        help="resume training",
    )

    main(parser.parse_args())

    # detected = model_loader.inference(
    #     inference_input=ObjectDetectionInferenceInput.from_urls([
    #         "https://photo.rever.vn/v3/get/rvk1YDxeNaQRcrIc_tsCaOeGFqwI1vrf5J7CPEnbZdIGTDnNnA7CAAG+iW3AJuMbBaE76MkTuDuQQF7M35VYC0Iw==/900x600/image.jpg",
    #         "https://photo.rever.vn/v3/get/rv3oj3knx5+QDK2tbxH+kshZeinoeWg6p1ugc5i9zyKon+6rpXp3Q6sP6u7+0VJHm+fo6vCH2PtzUdVMuIVeTq5g==/900x600/image.jpg"
    #     ]),
    #     save=True
    # )
