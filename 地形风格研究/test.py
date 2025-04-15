import numpy as np
import big_rocks 
import ores_in_rocks
import items_on_ground
import os
import msvcrt
import time
import sys
import platform

# 配置参数
VIEW_SIZE = 30    # 视口尺寸
HALF_VIEW = VIEW_SIZE // 2

# 初始化玩家
player_pos = [0, 0]  # 初始位于世界原点
player_char = '🧙'

# 获取输入 (Windows专用)
# 跨平台输入处理
def get_input():
    try:
        if platform.system() == 'Windows':
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch().decode().lower()
        else:
            import tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except:
        return None

def clear_input_buffer():
    if platform.system() == 'Windows':
        import msvcrt
        # 清空输入缓冲区
        while msvcrt.kbhit():
            msvcrt.getch()


# 清屏函数 (Windows专用)
def clear_screen():
    os.system('cls')

# 获取世界某位置的单元格
def get_world_cell(x, y):
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
    
    
# 获取当前视口
def get_viewport():
    viewport = np.empty((VIEW_SIZE, VIEW_SIZE), dtype=object)
    
    # 性能测试开始
    start_time = time.perf_counter()
    
    # 填充视口
    for y in range(VIEW_SIZE):
        for x in range(VIEW_SIZE):
            # 计算世界坐标
            world_x = player_pos[0] + x - HALF_VIEW
            world_y = player_pos[1] + y - HALF_VIEW
            
            # 获取单元格
            if x == HALF_VIEW and y == HALF_VIEW:
                viewport[y][x] = {'char': player_char, 'passable': True}
            else:
                viewport[y][x] = get_world_cell(world_x, world_y)
    
    # 性能测试结束
    elapsed = time.perf_counter() - start_time
    return viewport, elapsed

# 渲染游戏画面
def render():
    clear_screen()
    viewport, gen_time = get_viewport()
    
    for row in viewport:
        print(''.join([cell['char'] for cell in row]))
    
    # 显示性能信息
    print(f"玩家坐标: ({player_pos[0]}, {player_pos[1]})")
    print(f"地形生成时间: {gen_time*1000:.2f}ms")
    print("WASD移动 Q退出")

# 主游戏循环
while True:
    render()
    
    key = get_input()
    if key == 'q':
        break
    
    # 计算移动目标
    dx, dy = 0, 0
    if key == 'w': dy = -1
    elif key == 's': dy = 1
    elif key == 'a': dx = -1
    elif key == 'd': dx = 1
    
    new_x = player_pos[0] + dx
    new_y = player_pos[1] + dy
    
    # 检查目标位置是否可通行
    target_cell = get_world_cell(new_x, new_y)
    player_pos[0], player_pos[1] = new_x, new_y
    
    clear_input_buffer()
    time.sleep(0.05)

print("测试结束")