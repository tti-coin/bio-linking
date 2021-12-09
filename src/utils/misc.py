class ClosedInterval:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def __eq__(self, value):
        return self.min <= value <= self.max
    def __repr__(self):
        return f"[{self.min},{self.max}]"
