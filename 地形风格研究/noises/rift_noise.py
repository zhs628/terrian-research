'''  
. . . . . . . . . . . . . . . . . . . . # . . # . . . . . . . . . . . . # . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . # # # # . . . . . . . . . . . . # . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . # . . # # . . . . . . . . . . # . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . # . . . . # # . . . . . . . . # . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . # # # # # . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . # . . . . # # . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . #    
. . . . . . . . . . . . . . . . . . . . . # # . . . . . . . . . . . . . . . . . . . . . # .    
. . . . . . . . . . . . . . . . . . . . . # . # # . . . . . . . . . . . . . . . . . . . # .    
. . . . . . . . . . . . . . . . . . . . # . . . # # . . . . . . . . . . . . . . . . . # . .    
. . . . . . . # . . . . . . . . . . . # . . . . . . # . . . . . . . . . . . . . . . # . . .    
. . . . . . . . # . . . . . . . . . . . . . . . . . # . . . . . . . . . . . . . . # . . . .    
. . . . . . . . # . . . . . . . . . # . . . . . . . # # . . . . . . . . . . . . # # . . . .    
. . . . . . . . # # . . . . . . . # . . . . . . . . . # . . . . . . . . . . . . # . . . . .    
. . . . . . . # . . # # # . . . # . . . . . . . . . . # . . . . . . . . . . . # # . . . . .    
. . . . . . # # . . . . # # # # # # . . . . . . . . . # . . . . . . . . . . . # # . . . . .    
. . . . . . # . . . . . . . . . . . # # . . . . . . . # # . . . . . . . . . # # # # # . . .    
. . . . . . # . . . . . . . . . . . . . # # . . . . . # # # . . . . . . . # # . . . # # # #    
. . . . . . # . . . . . . . . . . . . . . . # # # # # . . # # # . . . # # # . . . . . . . #    
. . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . # # # # # . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . .    
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . .    
. # . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . .    
. . # # # # # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. # # . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. # . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
# . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
# . . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
. . . . . . . . . . # # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
'''

from noises.perlin import Perlin
from noises.voronoi import Voronoi 
import math
from vmath import vec2i
from array2d import array2d
from noises.utils import generate_layer_noise, apply_operations_area, filter_components_by_area


base_config = {'post_operations': [('Threshold(x, param)', 1.2)], 'seed': 123456}



layers_config = [  
                 {   'falloff': 55.46,
        'lacunarity': 0.0,
        'noise_type': 'Voronoi',
        'octaves': 1,
        'operations': [('Laplace(pos)', 0.0), ('Multiply(x, param)', -10.0)],
        'persistence': 0.0,
        'radius': 1,
        'scale': 10.89},
    {   'lacunarity': 0.0,
        'noise_type': 'Perlin',
        'octaves': 1,
        'operations': [('Threshold(x, param)', 0)],
        'persistence': 0.0,
        'scale': 20.66}
    ]



def noise_area(bottom_left: vec2i, width: float, height: float, step_size: int, seed: int) -> array2d[float]:
    voronoi = Voronoi(seed)
    perlin = Perlin(seed)
    
    # 计算基础网格参数
    cols = max(1, int(width / step_size))
    rows = max(1, int(height / step_size))

    # 创建坐标网格(使用array2d向量化操作)
    coords = array2d(cols + 2, rows + 2, default=None)
    for idx, _ in coords:
        coords[idx] = vec2i(
                    bottom_left.x + (idx.x - 1) * step_size,
                    bottom_left.y + (idx.y - 1) * step_size
                )

    # 初始化结果数组(带padding)
    result = array2d(cols, rows, 0)
    # 逐层生成噪声
    for layer in layers_config:
        # 生成当前层噪声(使用map向量化计算)
        padded_layer_noise = coords.map(lambda pos: generate_layer_noise(pos, layer, perlin=perlin, voronoi=voronoi))
        # 应用层操作(保持padding)
        layer_noise = apply_operations_area(padded_layer_noise, layer['operations'])
        # 累加到结果(使用array2d加法)
        result += layer_noise

    # 应用后处理并移除padding
    result = apply_operations_area(result, base_config['post_operations'])[1:-1, 1:-1]
    result = filter_components_by_area(result, 1, 'Moore', 10, 100)
    return result




# ==== 可视化 ====
if __name__ == '__main__':

    from array2d import array2d
    from noises.utils import terrain_to_ascii

    
    result = noise_area(vec2i(-0, 0), 80, 100, 1, base_config['seed'])


    print(terrain_to_ascii(result, size=(result.width, result.height)))
