import cv2
from datetime import datetime


from rml.model_loader.yolov8 import YOLOv8ModelLoader
from rml.domain.inference_input import ImageInferenceInput


model_loader = YOLOv8ModelLoader.from_pretrained(
    model_path='/Users/phamvy/Projects/source/local/object_detection/rml/data/models/best_35.pt',
    task=YOLOv8ModelLoader.DETECTION
    # model_path='best.pt'
    # model_path='rml/data/models/yolov8l-furniture.pt'
)

validation_data_dir = "/content/room_type/room_type/rever/val"
names = {
    0: "balcony",
    1: "bathroom",
    2: "bedroom",
    3: "dining_room",
    4: "empty_room",
    5: "hallway",
    6: "kitchen",
    7: "laundry_room",
    8: "living_room",
    9: "loggia",
    10: "stairs",
    11: "tabernacle_room",
    12: "terrace",
    13: "toilet",
    14: "undefined",
    15: "walkin_closet",
    16: "working_room"
}
mapping_ids = {
    0: 9,
    1: 0,
    2: 1,
    3: 4,
    4: -1,
    5: 3,
    6: 5,
    7: 6,
    8: 7,
    9: 9,
    10: 8,
    11: 12,
    12: 10,
    13: 0,
    14: 13,
    15: 2,
    16: 11
}

mapping_names = {
    0: "bathroom/toilet",
    1: "bedroom",
    2: "closet",
    3: "corridor",
    4: "dining_room",
    5: "kitchen",
    6: "laundry_room",
    7: "living_room",
    8: "staircase",
    9: "balcony",
    10: "terrace",
    11: "working_room",
    12: "tabernacle_room",
    13: "undefined"
}
# model_loader.validate(
#     validation_data_dir=validation_data_dir,
#     names=names,
#     mapping_ids=mapping_ids,
#     mapping_names=mapping_names
# )

# model_loader.export()
# metrics = model_loader.model.names
# print(metrics)

import os


detected = model_loader.inference(
    inference_input=ImageInferenceInput.from_urls([
        "https://images.unsplash.com/photo-1513694203232-719a280e022f?q=80&w=2938&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://photo.rever.vn/v3/get/rvsjhnHF4MpI_coUBbMsSaA2aMRIlmHQYIkejYuULxVaYEC2eJ5ENVGR2qmWSLZ653RNqCnCay6ehAtRX00xhcuQ==/900x600/image.jpg",
        "https://photo.rever.vn/v3/get/rvk1YDxeNaQRcrIc_tsCaOeGFqwI1vrf5J7CPEnbZdIGTDnNnA7CAAG+iW3AJuMbBaE76MkTuDuQQF7M35VYC0Iw==/900x600/image.jpg",
        "https://photo.rever.vn/v3/get/rv3oj3knx5+QDK2tbxH+kshZeinoeWg6p1ugc5i9zyKon+6rpXp3Q6sP6u7+0VJHm+fo6vCH2PtzUdVMuIVeTq5g==/900x600/image.jpg"
    ]),
    save=True
)


# start = datetime.now()
# detected = model_loader.inference(
#     inference_input=ObjectDetectionInferenceInput.from_paths([
#         "rml/data/20201226155223-7ad7_wm.jpg"
#     ]),
#     save=True
# )
# print(datetime.now() - start)
# model_loader.model = model_loader.export()
#
# detected = model_loader.inference(
#     inference_input=ObjectDetectionInferenceInput.from_urls([
#         "https://photo.rever.vn/v3/get/rvk1YDxeNaQRcrIc_tsCaOeGFqwI1vrf5J7CPEnbZdIGTDnNnA7CAAG+iW3AJuMbBaE76MkTuDuQQF7M35VYC0Iw==/900x600/image.jpg",
#         "https://photo.rever.vn/v3/get/rv3oj3knx5+QDK2tbxH+kshZeinoeWg6p1ugc5i9zyKon+6rpXp3Q6sP6u7+0VJHm+fo6vCH2PtzUdVMuIVeTq5g==/900x600/image.jpg"
#     ]),
#     save=True
# )

# from rml.vision.object_detection.models.yolov8.examples.YOLOv8_ONNXRuntime.main import YOLOv8
#
#
# start = datetime.now()
# img = YOLOv8(
#     onnx_model="rml/data/yolov8l-furniture.onnx",
#     input_image="rml/data/20201226155223-7ad7_wm.jpg",
#     confidence_thres=0.1,
#     iou_thres=0.1
# ).main()
# print(datetime.now() - start)
# import cv2
# cv2.imshow("image", img)
# cv2.waitKey(0)
#
#


