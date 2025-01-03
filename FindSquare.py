class FindSquare:
    def __init__(self, choppedARRAYS):
        self.choppedARRAYS = choppedARRAYS
        
    def plane_equation(self):
        x1, x2, x3 = self.choppedARRAYS[0][:3]
        y1, y2, y3 = self.choppedARRAYS[1][:3]
        z1, z2, z3 = self.choppedARRAYS[2][:3]

        AB = (x2 - x1, y2 - y1, z2 - z1)
        AC = (x3 - x1, y3 - y1, z3 - z1)

        A = AB[1] * AC[2] - AB[2] * AC[1]
        B = AB[2] * AC[0] - AB[0] * AC[2]
        C = AB[0] * AC[1] - AB[1] * AC[0]

        D = -(A * x1 + B * y1 + C * z1)

        return -A/C, -B/C, -C/C, -D/C

        
        