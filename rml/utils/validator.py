

class Score:
    pass


class ClassificationScore(Score):
    def __init__(self, accuracy: float, precision: float, recall: float):
        self.accuracy: float = accuracy
        self.precision: float = precision
        self.recall: float = recall


class Validator:
    def validate(self) -> Score:
        pass


class ClassificationValidator(Validator):
    def __init__(self, model):
        self.model = model
