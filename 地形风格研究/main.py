import streamlit as st
import numpy as np
from noises.perlin import Perlin
import matplotlib.pyplot as plt
import io
from datetime import datetime
import json
import pprint


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
st.set_page_config(layout="wide")





def terrain_to_ascii(terrain, chars=[".", "*", "#"], size=(40, 40)):
    """
    将地形图转换为ASCII艺术字符串

    参数：
    terrain -- 二维numpy数组，表示地形高度
    chars -- 字符列表，用于表示不同高度段（按从低到高顺序排列）
    size -- 可选元组(output_width, output_height)，控制输出尺寸

    返回：
    ASCII艺术字符串
    """
    if not chars:
        raise ValueError("字符列表不能为空")

    # 获取高度范围
    min_h = terrain.min()
    max_h = terrain.max()

    # 计算高度分段阈值
    num_chars = len(chars)
    thresholds = np.linspace(min_h, max_h, num_chars + 1)[1:-1]  # 中间阈值

    # 创建高度到字符的映射函数
    def height_to_char(h):
        for i, t in enumerate(thresholds):
            if h <= t:
                return chars[i]
        return chars[-1]

    # 处理采样尺寸
    original_shape = terrain.shape
    if size is None:
        # 默认保持原始比例
        scale = 1
    else:
        output_width, output_height = size
        # 计算采样间隔，保持宽高比
        x_scale = original_shape[1] / output_width if output_width else 1
        y_scale = original_shape[0] / output_height if output_height else 1
        scale = max(x_scale, y_scale)

    # 采样原始地形
    sampled_terrain = terrain[:: max(int(scale), 1), :: max(int(scale), 1)]

    # 转换为ASCII字符
    ascii_art = []
    for row in sampled_terrain:
        ascii_row = " ".join([height_to_char(h) for h in row])
        ascii_art.append(ascii_row)

    # 添加垂直填充保持比例
    if size is not None and output_height:
        target_lines = size[1]
        current_lines = len(ascii_art)
        if current_lines < target_lines:
            padding = np.random.choice(ascii_art, target_lines - current_lines)
            ascii_art += [str(p) for p in padding]

    return "\n".join(ascii_art)


# 初始化session状态
if "noise_layers" not in st.session_state:
    st.session_state.noise_layers = []
    st.session_state.world_size = 64
    st.session_state.combined = np.zeros(
        (st.session_state.world_size, st.session_state.world_size)
    )


# 噪声生成核心函数
def generate_perlin_noise(
    width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=123456
):
    world = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            world[x,y] = Perlin(seed=seed).noise_ex(
                x / scale,
                y / scale,
                0,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            # if x==10 and y==10:
            #     print(world[x,y], (
            #                     x / scale,
            #                     y / scale,
            #                     0,
            #                     octaves,
            #                     persistence,
            #                     lacunarity
            #                 )
            #           )
    return world


# 数学运算层处理器
def apply_math_operations(layer, operations):
    for op, param in operations:
        if op == "Power(x, param)":
            layer = np.power(layer, param)
        elif op == "Log(x, param)":
            layer = np.log(np.abs(layer) + 1e-9)
        elif op == "Sigmoid(x, param)":
            layer = 1 / (1 + np.exp(-param * layer))
        elif op == "Threshold(x, param)":
            layer = np.where(layer > param, 1.0, 0.0)
        elif op == "Add(x, param)":
            layer = layer + param
        elif op == "Multiply(x, param)":
            layer = layer * param
    return layer


# 界面布局
st.title("🌍 交互式柏林噪声生成器")

with st.sidebar:
    is_generating = st.button("✨ 生成噪声地图")
    # 基础参数设置
    with st.expander("⚙️ 全局参数", expanded=True):
        world_size = st.slider("世界尺寸", 64, 2048, 64, 32)
        num_layers = st.slider("噪声层数", 1, 8, 3)
        seed = int(st.text_input("seed (int)", "123456"))

    with st.expander("⚙️ 组合后处理", expanded=True):

        post_operation = st.selectbox(
            "后处理运算",
            [
                "None",
                "Power(x, param)",
                "Log(x, param)",
                "Sigmoid(x, param)",
                "Threshold(x, param)",
                "Add(x, param)",
                "Multiply(x, param)",
            ],
            key="post_op",
        )
        post_param = st.slider("后处理参数", -5.0, 5.0, 1.0, 0.1, key="post_param")

    # 图层参数生成
    layer_params = []
    for i in range(num_layers):
        with st.expander(f"🎚️ 第{i+1}层参数", expanded=True):
            helps = [
                """
scale​​:
控制噪声的缩放比例。
值越大，噪声的变化越平缓（低频噪声）；值越小，噪声的变化越剧烈（高频噪声）。
例如，scale=100.0 表示噪声的变化较为平滑，而 scale=10.0 表示噪声的变化更加细致。
""",
                """
​​octaves​​:
控制噪声的层数（或称为倍频）。
每一层噪声的频率和振幅会根据 lacunarity 和 persistence 进行调整。
值越大，噪声的细节越丰富，但计算量也会增加。
例如，octaves=6 表示生成 6 层不同频率的噪声。
""",
                """
​​persistence​​:
控制每一层噪声对最终结果的贡献程度（振幅的衰减因子）。
值越大，高频噪声的影响越强；值越小，高频噪声的影响越弱。
通常取值范围为 0.0 到 1.0。
例如，persistence=0.5 表示每一层噪声的振幅是前一层的一半。
""",
                """
​​lacunarity​​:
控制每一层噪声的频率增长倍数。
值越大，每一层噪声的频率增加得越快，细节越丰富。
通常取值大于 1.0。
例如，lacunarity=2.0 表示每一层噪声的频率是前一层的 2 倍。
""",
            ]

            in_use = st.checkbox(f"启用第{i+1}层", True)
            scale = st.slider(
                f"scale​​ {i+1}",
                10.0,
                500.0,
                100.0 * (i + 1),
                key=f"scale{i}",
                help=helps[0],
            )
            octaves = st.slider(
                f"​​octaves​​ {i+1}", 0, 10, 3, key=f"oct{i}", help=helps[1]
            )
            persistence = st.slider(
                f"​​persistence​​ {i+1}", 0.0, 2.0, 0.5, key=f"pers{i}", help=helps[2]
            )
            lacunarity = st.slider(
                f"​​lacunarity​​ {i+1}", 0.0, 4.0, 2.0, key=f"lac{i}", help=helps[3]
            )

            if f"layer_{i}_ops" not in st.session_state:
                st.session_state[f"layer_{i}_ops"] = []
            operations = st.session_state[f"layer_{i}_ops"]
            st.session_state[f"layer_{i}_ops"] = []
            # 显示已有操作
            num_op = st.slider(f"后处理数量{i}", 0, 10, 0, 1)
            for j in range(num_op):
                if not len(operations) <= j:
                    op = operations[j]
                else:
                    op = ["None", 0]

                cols = st.columns([3, 3, 1])
                with cols[0]:
                    new_op = st.selectbox(
                        f"操作{j+1}类型",
                        [
                            "None",
                            "Power(x, param)",
                            "Log(x, param)",
                            "Sigmoid(x, param)",
                            "Threshold(x, param)",
                            "Add(x, param)",
                            "Multiply(x, param)",
                        ],
                        index=(
                            0
                            if op[0] == "None"
                            else [
                                "Power(x, param)",
                                "Log(x, param)",
                                "Sigmoid(x, param)",
                                "Threshold(x, param)",
                                "Add(x, param)",
                                "Multiply(x, param)",
                            ].index(op[0])
                            + 1
                        ),
                        key=f"op_{i}_{j}",
                    )
                with cols[1]:
                    new_param = st.slider(
                        f"参数值", -5.0, 5.0, float(op[1]), key=f"param_{i}_{j}"
                    )
                op = [new_op, new_param]
                st.session_state[f"layer_{i}_ops"].append(op)

            if in_use:
                layer_params.append(
                    (
                        scale,
                        octaves,
                        persistence,
                        lacunarity,
                        st.session_state[f"layer_{i}_ops"],
                    )
                )

    # 生成并处理噪声
    if is_generating:
        st.session_state.world_size = world_size
        st.session_state.seed = seed
        raw_combined = np.zeros((world_size, world_size))
        noise_layers = []

        for i, params in enumerate(layer_params):
            scale, octaves, persistence, lacunarity, operations = params
            layer = generate_perlin_noise(
                world_size,
                world_size,
                scale=scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                seed=seed
            )
            layer = apply_math_operations(layer, operations)
            noise_layers.append(layer)
            raw_combined += layer  # 简单叠加

        st.session_state.noise_layers = noise_layers

        # 保存原始组合结果
        st.session_state.raw_combined = raw_combined

        st.session_state.combined = apply_math_operations(
            st.session_state.raw_combined, [(post_operation, post_param)]
        )
        st.session_state.layer_params = layer_params
        # print(st.session_state.combined[10,10])


if True:

    if st.session_state.noise_layers:
        st.subheader("可视化结果")

        # 添加控制 3D 视角和网格的滑动条
        with st.expander("🎚️ 3D 视图设置"):
            col_elev, col_azim, col_stride = st.columns(3)
            with col_elev:
                elev = st.slider("仰角 (elev)", 0, 90, 30, key="elev_slider")
            with col_azim:
                azim = st.slider("方位角 (azim)", 0, 90, 45, key="azim_slider")
            with col_stride:
                stride_scale = st.slider(
                    "网格规模", 2, 1000, 30, help="格子数量以及显示时的采样点数量"
                )
                stride = max(st.session_state.world_size // stride_scale, 1)

        with st.expander("图表尺寸和dpi"):
            col_w, col_h, col_dpi, col_hor = st.columns(4)
            with col_w:
                w = st.slider("宽度", 4, 48, 20, 1)
            with col_h:
                h = st.slider("高度", 4, 48, 16, 1)
            with col_dpi:
                dpi = st.slider("图表dpi", 10, 2000, 300, 1)
            with col_hor:
                if st.selectbox("布局方向", ["横向", "纵向"], 1) == "横向":
                    cr = 120
                else:
                    cr = 210

        # 添加切片控制选项
        with st.expander("🔪 切片视图设置"):
            slice_type = st.radio("切片方向", ["X轴切片", "Y轴切片"], horizontal=True)
            slice_pos = st.slider(
                "切片位置",
                0,
                st.session_state.world_size - 1,
                st.session_state.world_size // 2,
                help="选择要查看的切片位置",
            )

        # 代码生成控制
        with st.expander("👾 代码生成设置"):
            gen_ascii_terrian = st.checkbox("生成字符画", [True, False])
            ascii_terrian_chars = list(
                st.text_input("代表不同高度的字符(由低到高, 数量不限)", ".*#")
            )
            ascii_terrian_size = st.slider("字符画尺寸", 20, 300, 60, 20)

        tabs = st.tabs(
            [f"Layer {i+1}" for i in range(len(st.session_state.noise_layers))]
            + ["Combined"]
            + ["参数和代码"]
        )

        # 设置全局绘图样式
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "#0c1414"
        plt.rcParams["figure.facecolor"] = "#0c1414"
        plt.rcParams["savefig.facecolor"] = "#0c1414"
        plt.rcParams["text.color"] = "white"
        plt.rcParams["axes.labelcolor"] = "white"
        plt.rcParams["xtick.color"] = "white"
        plt.rcParams["ytick.color"] = "white"
        plt.rcParams["axes.edgecolor"] = "white"
        plt.rcParams["axes.titlecolor"] = "white"

        # 定义绘制切片图的函数
        def plot_slice(
            ax,
            data,
            slice_type,
            slice_pos,
            title="",
            color=None,
            linestyle="-",
            label=None,
        ):
            if slice_type == "X轴切片":
                slice_data = data[slice_pos, :]
                ax.plot(
                    np.arange(st.session_state.world_size),
                    slice_data,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                ax.set_title(f"{title} (X = {slice_pos})", color="white")
            else:
                slice_data = data[:, slice_pos]
                ax.plot(
                    np.arange(st.session_state.world_size),
                    slice_data,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                ax.set_title(f"{title} (Y = {slice_pos})", color="white")

            ax.set_facecolor("#0c1414")
            ax.tick_params(axis="both", colors="white")
            ax.grid(color="gray", linestyle=":", alpha=0.5)
            if label:
                ax.legend(facecolor="#0c1414", edgecolor="white")

        with tabs[-2]:  # 组合结果页
            fig = plt.figure(figsize=(w, h))
            # 3D视图
            ax1 = fig.add_subplot(cr + 1, projection="3d")

            # 应用网格采样
            X, Y = np.meshgrid(
                np.arange(0, st.session_state.combined.shape[1], stride),
                np.arange(0, st.session_state.combined.shape[0], stride),
            )
            Z = st.session_state.combined[::stride, ::stride]

            surf = ax1.plot_surface(
                X, Y, Z, rstride=1, cstride=1, cmap="rainbow", shade=False
            )
            ax1.view_init(elev=elev, azim=azim)
            if elev == 90:
                ax1.set_proj_type("ortho")
                ax1.set_box_aspect([1, 1, 0.001])

            ax1.set_title(
                f"Combined Layers 3D View (Sampling Distance: {stride})", color="white"
            )
            ax1.set_axis_off()
            fig.colorbar(surf, shrink=0.5, aspect=5, label="Intensity")

            # 切片视图 - 显示组合图层和所有单独图层
            ax2 = fig.add_subplot(cr + 2)

            plot_slice(
                ax2,
                st.session_state.raw_combined,
                slice_type,
                slice_pos,
                color="white",
                linestyle="--",
                label="Combined",
            )

            plot_slice(
                ax2,
                st.session_state.combined,
                slice_type,
                slice_pos,
                color="white",
                linestyle="-",
                label="Combined (Post Processed)",
            )

            # 再绘制各单独图层（虚线）
            colors = plt.cm.tab10.colors  # 使用tab10调色板获取不同颜色
            for i, layer in enumerate(st.session_state.noise_layers):
                plot_slice(
                    ax2,
                    layer,
                    slice_type,
                    slice_pos,
                    title="Combined Layers Slice",
                    linestyle="--",
                    label=f"Layer {i+1}",
                )

            st.pyplot(fig, dpi=dpi)

        for i, layer in enumerate(st.session_state.noise_layers):
            with tabs[i]:
                fig = plt.figure(figsize=(w, h))

                # 第一行：3D视图和切片视图
                ax1 = fig.add_subplot(221, projection="3d")

                # 应用网格采样
                X, Y = np.meshgrid(
                    np.arange(0, layer.shape[1], stride),
                    np.arange(0, layer.shape[0], stride),
                )
                Z = layer[::stride, ::stride]

                surf = ax1.plot_surface(
                    X, Y, Z, rstride=1, cstride=1, cmap="rainbow", shade=False
                )
                ax1.view_init(elev=elev, azim=azim)
                if elev == 90:
                    ax1.set_proj_type("ortho")
                    
                    ax1.set_box_aspect([1, 1, 0.001])
                ax1.set_title(
                    f"Layer {i+1} 3D View (Sampling Interval: {stride})", color="white"
                )
                ax1.set_axis_off()
                fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label="Intensity")

                # 切片视图
                ax2 = fig.add_subplot(222)
                plot_slice(ax2, layer, slice_type, slice_pos, f"Layer {i+1} Slice")

                # 第二行：直方图
                ax3 = fig.add_subplot(223)
                hist, bins = np.histogram(layer.flatten(), bins=50)
                ax3.bar(bins[:-1], hist, width=0.7 * (bins[1] - bins[0]), color="white")
                ax3.set_title("Value Distribution Histogram", color="white")
                ax3.tick_params(axis="both", colors="white")
                ax3.spines["bottom"].set_color("white")
                ax3.spines["top"].set_color("white")
                ax3.spines["left"].set_color("white")
                ax3.spines["right"].set_color("white")

                st.pyplot(fig, dpi=dpi)

        with tabs[-1]:

            layers_params = st.session_state.get("layer_params", default=layer_params)

            # 生成代码函数
            def generate_python_code():
                # 收集基础参数
                base_config = {
                    "post_operation": post_operation,
                    "post_param": post_param,
                    'seed': st.session_state.seed
                }

                # 收集各层参数
                layers_config = []

                for i, params in enumerate(layers_params):
                    layer_info = {
                        "scale": params[0],
                        "octaves(for pkpy)": params[1],
                        "persistence": params[2],
                        "lacunarity": params[3],
                        "operations": params[4],  # 直接存储操作列表
                    }
                    layers_config.append(layer_info)

                # 生成操作处理代码
                operations_code = ""
                for i, layer in enumerate(layers_config):
                    operations_code += f"""
                    # Layer {i+1} Operations
                    layer_operations = {json.dumps(layer['operations'])}"""

                # 构建代码模
                newline_char = "\n"
                code_template = f"""
'''  world_size:{st.session_state.world_size}x{st.session_state.world_size}, chars:{ascii_terrian_chars}
{terrain_to_ascii(st.session_state.combined, chars=ascii_terrian_chars, size=(ascii_terrian_size, ascii_terrian_size)) if gen_ascii_terrian else ''}
'''

from perlin import Perlin
import math
from linalg import vec2i


def apply_operations(x, operations):
    for op, param in operations:
        if op == 'Power(x, param)':
            x = x ** param
        elif op == 'Log(x, param)':
            x = math.log(abs(x) + 1e-9)  # 加上小常数以避免对0取对数
        elif op == 'Sigmoid(x, param)':
            x = 1 / (1 + math.exp(-param * x))
        elif op == 'Threshold(x, param)':
            x = 1.0 if x > param else 0.0
        elif op == 'Add(x, param)':
            x = x + param
        elif op == 'Multiply(x, param)':
            x = x * param
    return x  

base_config = {pprint.pformat(base_config, indent=4).replace('newline_char', '    ')}



layers_config = {pprint.pformat(layers_config, indent=4).replace('newline_char', '    ').replace('newline_char', '    ')}



def noise(v: vec2, seed=base_config['seed'])->float:
    '''对硬编码参数的Perlin噪声采样'''
    combined_noise = 0
    for layer in layers_config:
        scale, octaves, persistence, lacunarity, operations= (
            layer['scale'], layer['octaves(for pkpy)'], layer['persistence'],
            layer['lacunarity'], layer['operations']
        )
        noise = perlin_generator = Perlin(seed=seed).noise_ex(v.x/scale, v.y/scale, 0,
                    octaves,
                    persistence=persistence,
                    lacunarity=lacunarity)
        
        noise = apply_operations(noise, operations)
                
        combined_noise += noise

    combined_noise = apply_operations(combined_noise, [(base_config['post_operation'], base_config['post_param'])])
    return combined_noise

# ==== 可视化 ====
if __name__ == '__main__':

    from array2d import array2d
    import random
    
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
    
        return "\\n".join(ascii_art)
    
    result = array2d({st.session_state.world_size},{st.session_state.world_size}, default=noise)   # noise in rect(x:0, y:0, w:{st.session_state.world_size}, h:{st.session_state.world_size})
    
    print(terrain_to_ascii(result, size=(result.width, result.height)))
    
    
"""
                return code_template

            code_template = generate_python_code()

            st.code(code_template, "python", True)

            # 创建下载按钮
            code_blob = io.BytesIO()
            code_blob.write(code_template.encode("utf-8"))
            code_blob.seek(0)

            st.download_button(
                label=f"⬇️ 下载生成代码 (NoiseGenerator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py)",
                data=code_blob,
                file_name=f"NoiseGenerator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                mime="text/plain",
            )
            
            
