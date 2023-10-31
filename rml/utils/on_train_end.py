import os
import json
import shutil

import pandas as pd
import pathlib


class OnTrainEnd:
    RESULT_FILE = "results.csv"
    RESULT_FILE_IMG = "results.png"
    MODEL_FILE = "best.pt"
    BEST_MODEL_DIR = "best_model"

    def __init__(
            self,
            local_saved_dir: str,
            remote_saved_dir: str
    ):
        self.local_saved_dir: str = local_saved_dir
        self.remote_saved_dir: str = remote_saved_dir

    def on_train_end(self):
        self._prepare_remote_data(self.local_saved_dir, self.remote_saved_dir)

    def _prepare_remote_data(self, local_saved_dir: str, remote_saved_dir: str):
        pathlib.Path(os.path.join(remote_saved_dir, OnTrainEnd.BEST_MODEL_DIR)).mkdir(parents=True, exist_ok=True)
        # pathlib.Path(os.path.join(remote_saved_dir, "last_model")).mkdir(parents=True, exist_ok=True)

        results = pd.read_csv(os.path.join(local_saved_dir, "train", OnTrainEnd.RESULT_FILE))
        precision = results.iloc[0].values[4]
        recall = results.iloc[0].values[5]
        map = results.iloc[0].values[6]
        json.dump(
            {
                "precision": precision,
                "recall": recall,
                "map": map,
            },
            open(os.path.join(remote_saved_dir, OnTrainEnd.BEST_MODEL_DIR, "metrics.json"), "w")
        )

        shutil.copy(
            os.path.join(local_saved_dir, "train", OnTrainEnd.RESULT_FILE_IMG),
            os.path.join(remote_saved_dir, OnTrainEnd.BEST_MODEL_DIR, OnTrainEnd.RESULT_FILE_IMG)
        )
        shutil.copy(
            os.path.join(local_saved_dir, "train", "weights", OnTrainEnd.MODEL_FILE),
            os.path.join(remote_saved_dir, OnTrainEnd.BEST_MODEL_DIR, "model.pt")
        )




