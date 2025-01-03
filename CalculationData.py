import numpy as np


class CalculationData:
    def __init__(self, array) -> None:
        self.array = array

    def get_heights(self):
        x, y = self.array
        hist, xedges, yedges = np.histogram2d(x, y, bins=10, range=[
            [min(self.array[0]), max(self.array[0])],
            [min(self.array[1]), max(self.array[1])]
        ])

        xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()
        return xpos, ypos, zpos, dx, dy, dz
