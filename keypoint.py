class Keypoint:
    def __init__(self, pos_x: int, pos_y: int, radius: int, response: float):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.response = response

    def __repr__(self):
        return 'Keypoint(x: %d, y: %d, r: %d, response: %f)' % (self.pos_x, self.pos_y, self.radius, self.response)
