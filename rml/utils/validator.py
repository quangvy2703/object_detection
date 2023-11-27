

class Score:
    pass


class ClassificationScore(Score):
    def __init__(self, accuracy: float, precision: float, recall: float):
        self.accuracy: float = accuracy
        self.precision: float = precision
        self.recall: float = recall

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall
        }


class DetectionScore(Score):
    def __init__(self, accuracy: float, precision: float, recall: float):
        self.accuracy: float = accuracy
        self.precision: float = precision
        self.recall: float = recall

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall
        }
