"""
荒原

- 无状态
    - 山脉: Voronoi(荒原不是山地, 因此山脉比较稀疏)
        - 山体
        - 矿脉
        - 房间壁(外部 bit mask)

    - 裂谷: Voronoi

    - 水洼: 块状分布+零星小水坑
    
    - 枯木林: 稀疏的树木
    
    - 草丛: 块状分布+零星散布
    
    - 复杂结构(种子)
        - 小型废墟
        - 迷宫
    
- 有状态
    - 小型废墟
    - 迷宫

"""



import random
from typing import Callable, Literal
import base
from array2d import *
from typing import TypeVar, Generic
T = TypeVar('T')
from base import AsciiSprite, TilesetId, get_sprite
from vmath import color32, vec2i, vec2
import math
from noises import test_noise, rift_noise
from noises.voronoi import Voronoi

TileIndex = vec2i
 
# 配置参数
VIEW_SIZE = 100    # 视口尺寸

BASE_POSITION = vec2i(0,0)

SEED = 1234567

world: chunked_array2d[vec2i] = chunked_array2d(16)

SAMPLING_DISTANCE = 1

rnd = random.Random(SEED)


def sigmoid(x: float, a:float, b:float) -> float:
    return 1 / (1 + math.exp(-(a*x-b)))




def gradient(pos: vec2, noise_func: Callable[[vec2], float], d: float = 0.5) -> vec2:
    x, y = pos
    k = 1/(2 * d)
    k_diag = 1/(2 * d * math.sqrt(2))
    
    n_right = noise_func(vec2(x + d, y))
    n_left = noise_func(vec2(x - d, y))
    n_top = noise_func(vec2(x, y + d))
    n_bottom = noise_func(vec2(x, y - d))
    
    dx = (n_right - n_left) * k * 0.5
    dy = (n_top - n_bottom) * k * 0.5
    
    n_tr = noise_func(vec2(x + d, y + d))
    n_tl = noise_func(vec2(x - d, y + d))
    n_br = noise_func(vec2(x + d, y - d))
    n_bl = noise_func(vec2(x - d, y - d))
    
    dx += (n_tr - n_tl + n_br - n_bl) * k_diag * 0.5
    dy += (n_tr - n_br + n_tl - n_bl) * k_diag * 0.5
    
    return vec2(dx, dy)
    

def get_dir_tile(dir:vec2) -> TileIndex:
    # Calculate angle in [-π, π]
    angle = math.atan2(dir.y, dir.x)
    
    # Normalize to [0, 2π)
    if angle < 0:
        angle += 2 * math.pi
    
    # Define 8 directions (π/4 radians per sector)
    sector = int(round(angle / (math.pi / 4))) % 8
    
    # Map sector to arrow emoji
    arrows = [TileIndex(TilesetId.Direction, 2), # r
              TileIndex(TilesetId.Direction, 1), # ur
              TileIndex(TilesetId.Direction, 0), # u
              TileIndex(TilesetId.Direction, 7), # ul
              TileIndex(TilesetId.Direction, 6), # l
              TileIndex(TilesetId.Direction, 5), # dl
              TileIndex(TilesetId.Direction, 4), # d
              TileIndex(TilesetId.Direction, 3)  # dr
              ]
    return arrows[sector]

def get_contour_tile(height: float, height_range: tuple[float, float]) -> TileIndex:
    """
    根据高度值和范围获取等高线TileIndex
    
    参数:
        height: 当前高度值
        height_range: (min_height, max_height) 高度范围
        
    返回:
        TileIndex 对应等高线的瓦片索引
    """
    min_height, max_height = height_range
    
    # 计算高度在范围内的归一化值 [0, 1]
    normalized = (height - min_height) / (max_height - min_height)
    
    # 将归一化值映射到等高线瓦片索引 (假设有8个等高线等级)
    contour_levels = 8
    level = int(normalized * contour_levels)
    
    # 确保level在有效范围内
    level = max(0, min(contour_levels - 1, level))
    
    # 返回对应等高线的TileIndex (假设等高线瓦片在TilesetId.Height中连续排列)
    return TileIndex(TilesetId.Height, level)


def sample(array: array2d[T], distance: int) -> array2d[T]:
    # 确保采样后的数组大小不超过原始数组边界
    sampled_width = (array.width + distance - 1) // distance
    sampled_height = (array.height + distance - 1) // distance
    sampled_array = array2d(sampled_width, sampled_height)
    
    for x in range(0, min(array.width, sampled_width * distance), distance):
        for y in range(0, min(array.height, sampled_height * distance), distance):
            sampled_array[x // distance, y // distance] = array[x, y]
    return sampled_array
def threshold(x:float, threshold:float) -> Literal[1,0]:
    return 1 if x > threshold else 0


MIN = 10
MAX = -10
def compute_tile_on_(global_pos:vec2i) -> TileIndex:
    """计算一个位置的图块"""
    
    
    ground_noise, ground_noise_g = rift_noise.noise(global_pos, seed=SEED), gradient(global_pos, rift_noise.noise)
    gradient_value = ground_noise_g.length()
    
    global MIN, MAX
    MIN = min(MIN, ground_noise)
    MAX = max(MAX, ground_noise)
    
    # chasm
    # chasm_layer = TileIndex(TilesetId.Chasm, 0)
    
    # ground
    ground_layer = TileIndex(TilesetId.Ground, 0) 
    
    rift_layer = TileIndex(TilesetId.Rift, 0) if ground_noise else None
    
    
    grid_layer = TileIndex(TilesetId.Wall, 2) if global_pos.x % 30 == 0 and global_pos.y % 30 == 0 else None
    
    # overlay all layers
    layers = [
        # chasm_layer,  # 空
        ground_layer,
        grid_layer
    ]
    result = None
    for layer in layers[::-1]:
        if layer is not None:
            result = layer
            assert isinstance(result, TileIndex)
            return result
    
    
    raise ValueError("No layer found")
    
def compute_area(global_pos:vec2i, wh) -> TileIndex:
    

def generate_world_in_area(base: vec2i, wh: vec2i):
    world_area = world.view_rect(base, wh.x, wh.y)
    compute_points = sample(world_area, SAMPLING_DISTANCE)
    totals = compute_points.width * compute_points.height
    count = 0
    for pos, _ in compute_points:
        global_pos = pos * SAMPLING_DISTANCE + world_area.origin
        world[global_pos] = compute_tile_on_(global_pos)
        
        count += 1
        if count % 100 == 0:
            msg = str(round(count/totals*100, 2)) + '%     '
            print("\r" + msg, end="")
    print('')
        

def get_sprite_with_(tile_index: TileIndex|None) -> AsciiSprite:
    if tile_index == None:
        return AsciiSprite("❗")
    return get_sprite(tile_index.x, tile_index.y)

def render_world_view() -> array2d[AsciiSprite]:
    world_view = world.view_rect(BASE_POSITION, VIEW_SIZE, VIEW_SIZE)
    return array2d(VIEW_SIZE, VIEW_SIZE, lambda pos: get_sprite_with_(world_view[pos])) 
    
    
def tile_to_str(tile: AsciiSprite) -> None:
    """打印单字符"""
    char = tile.char
    
    if tile.bg is not None:
        char = color32.from_vec3i(tile.bg).ansi_bg(char)
    
    if tile.fg is not None:
        char = color32.from_vec3i(tile.fg).ansi_fg(char)
    
    print(char, end="")
    

def show_view(array: array2d[AsciiSprite]):
    width = array.width
    height = array.height
    
    print('-'*(width*2+2))
    
    for pos, tile in array[:, :]:
        
        if pos.x == 0:
            print('|', end='')
            
        tile_to_str(tile)
        
        if pos.x == width-1:
            print('|')
    
    print('-'*(width*2+2))


if __name__ == "__main__":


    generate_world_in_area(BASE_POSITION, vec2i(VIEW_SIZE, VIEW_SIZE))
    # print(world.view().render())
    show_view(sample(render_world_view(), SAMPLING_DISTANCE))
    print(MIN, MAX)
