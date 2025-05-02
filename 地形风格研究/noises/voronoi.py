from random import Random
import math
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(t, a, b):
    return a + t * (b - a)

def hash3f(x: int, y: int, z: int) -> tuple[float, float, float]:
    # 计算初始哈希，避免 n=0 和负数问题
    n = (abs(x) * 73856093 ^ abs(y) * 19349663 ^ abs(z) * 83492791) & 0x7fffffff
    n = max(n, 1)  # 确保 n ≥ 1

    # 生成伪随机序列
    rx = (n * 16807) % 2147483647
    ry = (rx * 16807) % 2147483647
    rz = (ry * 16807) % 2147483647

    # 防御性检查：确保 rz ≠ 0
    if rz == 0:
        rz = 2147483646

    # 归一化到 [0, 1)
    return rx / 2147483647, ry / 2147483647, rz / 2147483647



class Voronoi:
    def __init__(self, seed: int | None = None):
        rnd = Random(seed)
        permutation = list(range(
            256))
        rnd.shuffle(permutation)
        self.p = permutation * 2

        self.vecs = [
            (rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))
            for _ in range(256)
        ]

    def grad(self, hash: int, x: float, y: float, z: float):
        v = self.vecs[hash]     # hash: [0, 255]
        return v[0] * x + v[1] * y + v[2] * z
    

    
    
    def noise_ex(self, x: float, y: float, z: float, radius:int, falloff:float, octaves: int, persistence=0.5, lacunarity=2.0):
        total = 0.0
        frequency = 1
        amplitude = 1
        max_value = 0
        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency, z * frequency,radius, falloff) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return total / max_value

    def noise(self, x: float, y: float, z: float, radius=1, falloff=8.0) -> float:
        # 晶格坐标和局部坐标
        p = (int(math.floor(x)), int(math.floor(y)), int(math.floor(z)))
        f = (x - p[0], y - p[1], z - p[2])  # fract(x,y,z)
    
        res = 0.0
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    # 当前晶格点 (i,j,k)
                    b = (p[0] + di, p[1] + dj, p[2] + dk)
                    # 用 hash3f 生成随机偏移 (rx, ry, rz)
                    rx, ry, rz = hash3f(*b)
                    # 计算相对向量 r = (i,j,k) - f + random_offset
                    r = (di - f[0] + rx, dj - f[1] + ry, dk - f[2] + rz)
                    # 计算距离平方 d = r·r
                    d = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
                    # 贡献 = 1/d⁸（平滑衰减）
                    epsilon = 1e-10  
                    res += 1.0 / (max(d, epsilon) ** falloff)
    
        # 归一化 (1/res)^(1/16)
        return (1.0 / res) ** (1.0 / 16.0)