"""
rever-sagemaker
    ai
        project-name-dir
            best-model-dir
                training_info.json
                    training_time: datetime
                    duration: datetime
                    instance_info:
                        py_version: str = "py38",
                        instance_count: int = 1,
                        instance_type: str = "ml.g4dn.xlarge",
                    training_data_info:
                        source: str
                        data_dir: str
                        num_samples: int
                        num_classes: int
                model-checkpoint.pt
                validation-score.json
                    ap: float
                    precision: float
                    recall: float
            last-model-dir
                training-info.json
                model-checkpoint.pt
                validation-score.json

"""

