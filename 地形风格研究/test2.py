from typing import Callable, Literal
import base
from array2d import *
from typing import TypeVar, Generic
T = TypeVar('T')
from base import AsciiSprite, TilesetId, get_sprite
from vmath import color32, vec2i, vec2
import math
from noises import test_noise

TileIndex = vec2i
 
# 配置参数
VIEW_SIZE = 50    # 视口尺寸

BASE_POSITION = vec2i(0,0)

SEED = 123456

world: chunked_array2d[vec2i] = chunked_array2d(16)

SAMPLING_DISTANCE = 1





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

def compute_tile_on_(global_pos:vec2i) -> TileIndex:
    """计算一个位置的图块"""
    
    # chasm
    chasm_layer = TileIndex(TilesetId.Chasm, 0)
    
    # ground
    ground_noise, ground_noise_g = test_noise.noise(global_pos, seed=SEED), gradient(global_pos, test_noise.noise)
    is_higher_ground_area = threshold(ground_noise, 0.075)
    is_ground_area = threshold(ground_noise, 0.03)
    is_ground_edge_area = is_ground_area and not is_higher_ground_area
    
    ground_layer = TileIndex(TilesetId.Ground, 2) if is_ground_area else None
    higher_ground_layer = TileIndex(TilesetId.Ground, 3) if is_higher_ground_area else None
    
    gradient_value = ground_noise_g.length()
    
    # beach
    is_plat = not threshold(gradient_value, 0.02) and is_ground_area
    # is_plat = is_ground_area
    is_beach = is_plat and is_ground_edge_area
    
    plat_beach_layer = TileIndex(TilesetId.Tree, 2) if is_beach else None
    
    
    # cliff (hard to pass)
    is_cliff = threshold(gradient_value, 0.012) and is_ground_area
    
    cliff_layer = TileIndex (TilesetId.Wall, 1) if is_cliff else None
    
    
    # overlay all layers
    layers = [
        chasm_layer,  # 空
        ground_layer,  # 浅黄色地面 (海平面以上)
        higher_ground_layer,  # 柠檬黄地面  (海拔更高的地方)
        plat_beach_layer,  # 椰子树  (海平面以上, 海拔低, 坡度小)
        # cliff_layer  # 石头 (海平面以上, 坡度大)
    ]
    result = None
    for layer in layers[::-1]:
        if layer is not None:
            result = layer
            return result
    
    if result is None:
        raise ValueError("No layer found")
    
    
    

def generate_world_in_area(base: vec2i, wh: vec2i):
    world_area = world.view_rect(base, wh.x, wh.y)
    for base, _ in world_area:
        global_pos = base + world_area.origin
        world[global_pos] = compute_tile_on_(global_pos)
    
def get_sprite_with_(tile_index: TileIndex) -> AsciiSprite:
    return get_sprite(tile_index.x, tile_index.y)

def render_world_view() -> array2d[AsciiSprite]:
    world_view = world.view_rect(BASE_POSITION, VIEW_SIZE, VIEW_SIZE)
    return array2d(VIEW_SIZE, VIEW_SIZE, lambda pos: get_sprite_with_(world_view[pos])) 
    
    
def print_tile(tile: AsciiSprite) -> None:
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
            
        print_tile(tile)
        
        if pos.x == width-1:
            print('|')
    
    print('-'*(width*2+2))


if __name__ == "__main__":
    generate_world_in_area(BASE_POSITION, vec2i(VIEW_SIZE, VIEW_SIZE))
    # print(world.view().render())
    show_view(sample(render_world_view(), SAMPLING_DISTANCE))