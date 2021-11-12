class Keypoint:
    def __init__(self, pos_x: int, pos_y: int, size: int, response: float, angle: float = None):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.size = size
        self.response = response
        self.angle = angle

    def __repr__(self):
        return 'Keypoint(x: %d, y: %d, r: %d, response: %f, angle: %f)' % (
        self.pos_x, self.pos_y, self.size, self.response, self.angle)
