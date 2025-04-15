from perlin import Perlin
from array2d import *
from base import AsciiSprite, TilesetId
from linalg import vec2i

# 配置参数
VIEW_SIZE = 30    # 视口尺寸

base_pos = vec2i(0,0)

world = chunked_array2d(16)


# 计算入口
def compute(v:vec2i) -> AsciiSprite:
    pass

# 第一步计算出tileset
def compute_tileset(v:vec2i) -> TilesetId:
    val = big_rocks.noise(x, y)
    result = None
    if val < 0.5:
        result = {'char': '🟨'} 
    else:
        result = {'char': '🟫'}
    
    val = ores_in_rocks.noise(x, y)
    
    if val > 0.5:
        result = {'char': '💎'} 
        
    val = items_on_ground.noise(x, y)
    
    if val > 0.5:
        result = {'char': '🌵'} 

    return result

def get_viewport() -> array2d[AsciiSprite]:
    viewport = world.view_rect(base_pos, VIEW_SIZE, VIEW_SIZE)\
    viewport.

def render(array: array2d[str]):
    
    