from typing import Optional


class Keypoint:
    def __init__(self, pos_x: int, pos_y: int, octave: int, angle: Optional[float], response: float, layer: int,
                 size: float):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.octave = octave
        self.layer = layer
        self.angle = angle
        self.response = response
        self.size = size

    def get_input_image_pos_x(self):
        return int((self.pos_x + (2 ** self.octave)))

    def get_input_image_pos_y(self):
        return int((self.pos_y + (2 ** self.octave)))

    def get_input_size(self):
        return self.size * 2 ** (self.octave + 1)

    def __hash__(self):
        return hash((self.pos_x, self.pos_y, self.angle, self.response, self.size))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented

        return self.get_input_image_pos_x() == other.get_input_image_pos_x() \
               and self.get_input_image_pos_y() == other.get_input_image_pos_y() \
               and self.angle == other.angle \
               and self.response == other.response \
               and self.get_input_size() == other.get_input_size()


def remove_duplitcates(keypoints: list[Keypoint]):
    return list(set(keypoints))
