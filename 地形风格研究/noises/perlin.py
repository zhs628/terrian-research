from random import Random

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(t, a, b):
    return a + t * (b - a)

class Perlin:
    def __init__(self, seed: int | None = None):
        rnd = Random(seed)
        permutation = list(range(256))
        rnd.shuffle(permutation)
        self.p = permutation * 2

        self.vecs = [
            (rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))
            for _ in range(256)
        ]

    def grad(self, hash: int, x: float, y: float, z: float):
        v = self.vecs[hash]     # hash: [0, 255]
        return v[0] * x + v[1] * y + v[2] * z
    
    def noise_ex(self, x: float, y: float, z: float, octaves: int, persistence=0.5, lacunarity=2.0):
        total = 0.0
        frequency = 1
        amplitude = 1
        max_value = 0
        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency, z * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return total / max_value

    def noise(self, x: float, y: float, z: float):
        p = self.p
        # find the unit cube that contains the point
        X, Y, Z = int(x) & 255, int(y) & 255, int(z) & 255
        # find relative x, y, z of point in cube
        x, y, z = x - int(x), y - int(y), z - int(z)
        # compute fade curves for x, y, z
        u, v, w = fade(x), fade(y), fade(z)
        # hash coordinates of the 8 cube corners
        _0 = p[p[X  ] + Y  ] + Z
        _2 = p[p[X+1] + Y  ] + Z
        _1 = p[p[X  ] + Y+1] + Z
        _3 = p[p[X+1] + Y+1] + Z
        """
        _0: h(X, Y)      _2: h(X+1, Y)
        *----------------*
        |                |
        |    *           |
        |    (x, y)      |
        |                |
        |                |
        |                |
        *----------------*
        _1: h(X, Y+1)    _3: h(X+1, Y+1)
        """
        # add blended results from 8 corners of cube
        return lerp(w, lerp(v, lerp(u, self.grad(p[_0  ], x  , y  , z   ),
                                       self.grad(p[_2  ], x-1, y  , z   )),
                               lerp(u, self.grad(p[_1  ], x  , y-1, z   ),
                                       self.grad(p[_3  ], x-1, y-1, z   ))),
                       lerp(v, lerp(u, self.grad(p[_0+1], x  , y  , z-1 ),
                                       self.grad(p[_2+1], x-1, y  , z-1 )),
                               lerp(u, self.grad(p[_1+1], x  , y-1, z-1 ),
                                       self.grad(p[_3+1], x-1, y-1, z-1 ))))