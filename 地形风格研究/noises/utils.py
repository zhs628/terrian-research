
from .perlin import Perlin
from .voronoi import Voronoi 
import math
from vmath import vec2i

import math
from typing import Callable, Tuple, Union

from array2d import array2d, array2d_like
import random
def convolve3x3_trimmed_float(
    input: array2d_like[float], 
    kernel: array2d_like[float]
) -> array2d_like[float]:
    """
    Perform 3x3 convolution with float precision, discarding outer edges.
    Ensures input and kernel are converted to float if not already.
    
    Args:
        input: Input array (H, W)
        kernel: 3x3 convolution kernel
        
    Returns:
        Convolved array of shape (H-2, W-2), with float dtype.
    """
    input = input.map(float)  # Convert entire array to float
    kernel = kernel.map(float)  # Convert kernel to float
    
    H, W = input.n_rows, input.n_cols
    
    output = array2d(W - 2, H - 2)
    
    k00, k01, k02 = kernel[0,0], kernel[0,1], kernel[0,2]
    k10, k11, k12 = kernel[1,0], kernel[1,1], kernel[1,2]
    k20, k21, k22 = kernel[2,0], kernel[2,1], kernel[2,2]
    
    for i in range(W - 2):  
        for j in range(H - 2): 
            output[i, j] = (
                input[i,   j] * k00 + input[i,   j+1] * k01 + input[i,   j+2] * k02 +\
                input[i+1, j] * k10 + input[i+1, j+1] * k11 + input[i+1, j+2] * k12 +\
                input[i+2, j] * k20 + input[i+2, j+1] * k21 + input[i+2, j+2] * k22
            )
    
    return output



def apply_operations_area(padded_data: array2d[float],
                         operations: list[Tuple[str, float]]) -> array2d[float]:
    '''
    Returns the result of applying a list of operations to a 2D array of floats.
    * result.size == padded_data[p:-p, p:-p].size
    * p = sum(op in ['Laplace(pos)', 'GradientMagnitude(pos)'] for op in operations)
    '''
    result = padded_data.copy()
    min_shape_x = result.shape.x - 2

    for op, param in operations:
        if op == 'Power(x, param)':
            result = result ** param
        elif op == 'Power(param, x)':
            result = param ** result
        elif op == 'Log(x, param)':
            result = result.map(lambda x: math.log(abs(x) + 1e-9))
        elif op == 'Sigmoid(x, param)':
            result = result.map(lambda x: 1 / (1 + math.exp(-param * x)))
        elif op == 'Threshold(x, param)':
            result = (result > param).map(lambda x: 1 if x else 0)
        elif op == 'Add(x, param)':
            result = result + param
        elif op == 'Multiply(x, param)':
            result = result * param
        elif op == 'Laplace(pos)':
            result = laplace_area(result)
            min_shape_x = min(min_shape_x, result.shape.x)
        elif op == 'GradientMagnitude(pos)':
            result = gradient_magnitude_area(result)
            min_shape_x = min(min_shape_x, result.shape.x)

    padding = (result.shape.x - min_shape_x)//2
    if padding == 0:
        return result
    return result[padding:-padding, padding:-padding]

def generate_layer_noise(pos: vec2i, layer: dict, perlin:Perlin|None = None, voronoi:Voronoi|None = None) -> float:
    if layer['noise_type'] == 'Perlin':
        return perlin.noise_ex(
            pos.x/layer['scale'], pos.y/layer['scale'], 0,
            layer['octaves'],
            persistence=layer['persistence'],
            lacunarity=layer['lacunarity'])
    elif layer['noise_type'] == 'Voronoi':
        return voronoi.noise_ex(
            pos.x/layer['scale'], pos.y/layer['scale'], 0,
            layer['radius'],
            layer['falloff'],
            layer['octaves'],
            persistence=layer['persistence'],
            lacunarity=layer['lacunarity'])

def laplace_area(padded_data: array2d[float]) -> array2d[float]:
    '''
    Returns the Laplace operator applied to a 2D array.(result.size == padded_data[1:-1, 1:-1].size)
    '''
    kernel = type(padded_data).fromlist([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    return convolve3x3_trimmed_float(padded_data, kernel)


def gradient_magnitude_area(padded_data: array2d[float]) -> array2d[float]:
    kernel_x = type(padded_data).fromlist(
        [[-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]])
    kernel_y = type(padded_data).fromlist(
        [[-1, -2, -1], 
        [0, 0, 0], 
        [1, 2, 1]])
    
    return (convolve3x3_trimmed_float(padded_data, kernel_x) + convolve3x3_trimmed_float(padded_data, kernel_y)) ** 0.5

def terrain_to_ascii(terrain, chars=(".", "*", "#"), size=(40, 40)):

    if not chars:
        raise ValueError("字符列表不能为空")

    # 获取地形数据的最小和最大值
    max_h_pos, max_h = max(terrain, key=lambda pos_value_pair: pos_value_pair[1])
    min_h_pos, min_h = min(terrain, key=lambda pos_value_pair: pos_value_pair[1])

    if min_h is None or max_h is None:
        return ""  # 空地形

    # 计算高度分段阈值
    num_chars = len(chars)
    if num_chars == 0:
        raise ValueError("字符列表不能为空")
    thresholds = []
    for i in range(1, num_chars):
        fraction = i / num_chars
        thresholds.append(min_h + fraction * (max_h - min_h))

    # 创建高度到字符的映射函数
    def height_to_char(h):
        for i, t in enumerate(thresholds):
            if h <= t:
                return chars[i]
        return chars[-1]

    # 处理采样尺寸
    original_cols = terrain.n_cols
    original_rows = terrain.n_rows
    step = 1
    if size is not None:
        output_width, output_height = size
        if output_width <= 0 or output_height <= 0:
            raise ValueError("输出尺寸必须大于0")
        x_scale = original_cols / output_width
        y_scale = original_rows / output_height
        scale = max(x_scale, y_scale)
        step = max(int(scale), 1)

    # 采样原始地形
    sampled_terrain = terrain[::step, ::step]
    sampled_data = sampled_terrain.tolist()

    # 转换为ASCII字符
    ascii_art = []
    for row in sampled_data:
        ascii_row = " ".join([height_to_char(h) for h in row])
        ascii_art.append(ascii_row)

    # 添加垂直填充保持比例
    if size is not None:
        target_width, target_height = size
        current_lines = len(ascii_art)
        if current_lines < target_height:
            padding_needed = target_height - current_lines
            padding = random.choices(ascii_art, k=padding_needed)
            ascii_art.extend(padding)

    return "\n".join(ascii_art)

def filter_components_by_area(
    data: array2d, 
    value,
    neighborhood_mode: str, 
    min_area: int,
    max_area: int 
) -> array2d:
    """
    将连通域面积在 [min_area, max_area] 范围内的区域保留，其余设为 0。
    
    Args:
        data: 输入的二维数组
        value: 目标值（要找的连通域的值）
        neighborhood_mode: 邻域类型（4-邻域或8-邻域）
        min_area: 最小连通域面积（含）
        max_area: 最大连通域面积（含），默认为无穷大，即不设上限
        
    Returns:
        处理后的数组，只保留面积在指定范围内的连通域，其余设为 0
    """
    # Step 1: 获取连通域标记
    visited, count = data.get_connected_components(value, neighborhood_mode)
    
    # Step 2: 统计每个连通域的面积
    area = {}  # component_id -> pixel count
    for i in range(visited.n_cols):
        for j in range(visited.n_rows):
            component_id = visited[i, j]
            if component_id > 0:
                area[component_id] = area.get(component_id, 0) + 1
    
    # Step 3: 找出要保留的连通域（面积在 [min_area, max_area] 范围内）
    valid_components = {cid for cid, cnt in area.items() if min_area <= cnt <= max_area}
    
    # Step 4: 生成结果数组
    result = data.copy()
    mask = visited.map(lambda x: 1 if x in valid_components else 0)
    result = result * mask
    
    return result
