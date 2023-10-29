class InferenceResult:
    pass


class Box:
    pass


class OIBox(Box):
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min: float = x_min
        self.y_min: float = y_min
        self.x_max: float = x_max
        self.y_max: float = y_max

    @staticmethod
    def from_dict(data: dict):
        return OIBox(
            x_min=data["XMin"],
            y_min=data["YMin"],
            x_max=data["XMax"],
            y_max=data["YMax"]
        )


class COCOBox(Box):
    def __init__(self, x_center: float, y_center: float, width: float, height: float):
        self.x_center: float = x_center
        self.y_center: float = y_center
        self.width: float = width
        self.height: float = height

    @staticmethod
    def from_oi_box(oi_box: OIBox):
        width = oi_box.x_max - oi_box.x_min
        height = oi_box.y_max - oi_box.y_min
        return COCOBox(
            x_center=oi_box.x_min + width / 2,
            y_center=oi_box.y_min + height / 2,
            width=width,
            height=height
        )

    @staticmethod
    def from_dict(data: dict):
        return COCOBox(
            x_center=data["x_center"],
            y_center=data["y_center"],
            width=data["width"],
            height=data["height"]
        )
