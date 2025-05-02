from anyio import key
import noise
import streamlit as st
import numpy as np
from noises.perlin import Perlin
from noises.voronoi import Voronoi
import matplotlib.pyplot as plt
import io
from datetime import datetime
import json
import pprint
from scipy.ndimage import laplace
from scipy.signal import convolve2d
import pickle


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
st.set_page_config(layout="wide")

def float_distribution(arr, bins=10, range=None):
    # 将二维数组展平为一维
    flattened = np.array(arr).flatten()
    # 计算直方图
    counts, bin_edges = np.histogram(flattened, bins=bins, range=range)
    # 返回桶边界和计数
    return bin_edges, counts


def nested_hash(obj):
    """
    生成嵌套结构的哈希值，处理字典、列表和基本对象
    """
    if isinstance(obj, (int, float, str, bool, bytes)) or obj is None:
        return hash(obj)
    elif isinstance(obj, (tuple, frozenset)):
        return hash(tuple(nested_hash(e) for e in obj))
    elif isinstance(obj, list):
        return hash(tuple(nested_hash(e) for e in obj))
    elif isinstance(obj, dict):
        return hash(tuple(sorted((k, nested_hash(v)) for k, v in obj.items())))
    elif hasattr(obj, "__dict__"):
        return nested_hash(vars(obj))
    else:
        raise TypeError(f"不可哈希类型: {type(obj)}")


def get_param_state_hash():
    return nested_hash(get_param_state())


def get_param_state():
    if "param" not in st.session_state:
        st.session_state.param = {}
        st.session_state.param["world_size"] = 64
        st.session_state.param["seed"] = 123456
        st.session_state.param["layer_params"] = []
        st.session_state.param["post_operations"] = []
    return st.session_state.param


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


# 噪声生成核心函数
def generate_perlin_noise(
    width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=123456
):
    world = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            world[x, y] = Perlin(seed=seed).noise_ex(
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


def generate_voronoi_noise(
    width,
    height,
    scale=100.0,
    radius=1,
    falloff=8.0,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    seed=123456,
):
    world = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            world[x, y] = Voronoi(seed=seed).noise_ex(
                x / scale,
                y / scale,
                0,
                radius=radius,
                falloff=falloff,
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
def apply_math_operations(padded_data, operations):
    padded_data = padded_data.copy()
    for op, param in operations:
        if op == "Power(x, param)":
            padded_data = np.power(padded_data, param)
        elif op == "Power(param, x)":
            padded_data = np.power(param, padded_data)
        elif op == "Log(x, param)":
            padded_data = np.log(np.abs(padded_data) + 1e-9)
        elif op == "Sigmoid(x, param)":
            padded_data = 1 / (1 + np.exp(-param * padded_data))
        elif op == "Threshold(x, param)":
            padded_data = np.where(padded_data > param, 1.0, 0.0)
        elif op == "Add(x, param)":
            padded_data = padded_data + param
        elif op == "Multiply(x, param)":
            padded_data = padded_data * param
        elif op == "Laplace(pos)":
            padded_data = laplace(padded_data)
        elif op == 'GradientMagnitude(pos)':
            kernel_x = np.array([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]])
            gradient_x = convolve2d(padded_data, kernel_x, mode='same', boundary='symm')
            gradient_y = convolve2d(padded_data, kernel_y, mode='same', boundary='symm')
            padded_data = np.sqrt(gradient_x**2 + gradient_y**2)
        else:
            raise ValueError(f"未知运算符: {op}")
    return padded_data[1:-1, 1:-1]


# 界面布局
st.title("🌍 交互式柏林噪声生成器")

with st.sidebar:

    
    with st.expander("💾 网页缓存"):
        pkl = st.file_uploader("恢复网页cache")
        if pkl:
            d = pickle.loads(pkl.read())
            st.toast(d, icon="💾")
            for k,v in d.items():
                st.session_state[k] = v
        
        st.download_button("保存网页cache", pickle.dumps(st.session_state.to_dict()), f'st_cached_{datetime.now()}.pkl')
    
    is_generating = False
    is_generating = st.button("✨ 生成噪声地图")

    # 基础参数设置
    with st.expander("⚙️ 全局参数", expanded=True):
        world_size = st.slider("世界尺寸", 64, 2048, 64, 32, key="world_size")
        num_layers = st.slider("噪声层数", 1, 8, 1, key="num_layers")
        seed = int(st.text_input("seed (int)", "123456", key="seed"))

        if st.checkbox("启用全局后处理", False, key="enable_post_processing"):
            op = st.selectbox(
                "后处理运算",
                [
                    "Power(x, param)",
                    "Power(param, x)",
                    "Log(x, param)",
                    "Sigmoid(x, param)",
                    "Threshold(x, param)",
                    "Add(x, param)",
                    "Multiply(x, param)",
                    "Laplace(pos)",
                    "GradientMagnitude(pos)"
                ],
                key="post_op",
            )
            param = st.number_input("后处理参数", value=1.0, key="post_param")
            post_operations = [(op, param)]
        else:
            post_operations = []

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

            noise_type = st.selectbox(
                f"噪声类型{i+1}",
                ["Perlin", "Voronoi"],
                key=f"noise_type{i}",
            )
            radius = None
            falloff = None
            if noise_type == "Voronoi":
                radius = st.slider(
                    f"radius {i+1}",
                    0,
                    10,
                    1,
                    key=f"radius{i}",
                    help="计算每一个点时考虑周围的晶格点半径(radius=1时考虑3x3), 同时直接影响耗时",
                )
                falloff = st.slider(
                    f"falloff {i+1}",
                    1.0,
                    100.0,
                    1.0,
                    key=f"falloff{i}",
                    help="晶格距离对高度贡献权重的衰减因子, 越大代表着近处的晶格越是重要",
                )

            scale = st.slider(
                f"scale​​ {i+1}",
                10.0,
                500.0,
                100.0 * (i + 1),
                key=f"scale{i}",
                help=helps[0],
            )
            octaves = st.slider(
                f"​​octaves​​ {i+1}", 1, 10, 3, key=f"oct{i}", help=helps[1]
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
            num_op = st.slider(f"后处理数量{i}", 0, 10, 0, 1, key=f"num_op_{i}")
            ops = []
            for j in range(num_op):

                cols = st.columns([3, 3, 1])
                with cols[0]:
                    new_op = st.selectbox(
                        f"操作{i}_{j+1}类型",
                        [
                            "Power(x, param)",
                            "Power(param, x)",
                            "Log(x, param)",
                            "Sigmoid(x, param)",
                            "Threshold(x, param)",
                            "Add(x, param)",
                            "Multiply(x, param)",
                            "Laplace(pos)",
                            "GradientMagnitude(pos)"
                        ],
                        key=f"op_{i}_{j}",
                    )
                with cols[1]:
                    new_param = st.number_input(f"操作{i}_{j+1} 参数", value=1.0, key=f"param_{i}_{j}")
                ops.append((new_op, new_param))
                
            st.session_state[f"layer_{i}_ops"] = ops

            if in_use:
                if noise_type == "Perlin":
                    layer_params.append(
                        {
                            "noise_type": noise_type,
                            "scale": scale,
                            "octaves": octaves,
                            "persistence": persistence,
                            "lacunarity": lacunarity,
                            "operations": st.session_state[f"layer_{i}_ops"],
                        }
                    )
                elif noise_type == "Voronoi":

                    layer_params.append(
                        {
                            "radius": radius,
                            "falloff": falloff,
                            "noise_type": noise_type,
                            "scale": scale,
                            "octaves": octaves,
                            "persistence": persistence,
                            "lacunarity": lacunarity,
                            "operations": st.session_state[f"layer_{i}_ops"],
                        }
                    )
                else:
                    raise ValueError(f"噪声类型错误: {noise_type}")

    now_param = {
        "world_size": world_size,
        "seed": seed,
        "post_operations": post_operations,
        "layer_params": layer_params,
    }
    st.write(now_param)
    st.session_state.param_hash = nested_hash(now_param)


    # 生成并处理噪声
    if is_generating:
        
        for k, v in now_param.items():
            get_param_state()[k] = v
            


        noise_layers = []
        raw_combined = None

        for i, params in enumerate(layer_params):
            
            noise_type = params["noise_type"]
            scale, octaves, persistence, lacunarity, operations = (
                params["scale"],
                params["octaves"],
                params["persistence"],
                params["lacunarity"],
                params["operations"],
            )

            padded_noise = None
            if noise_type == "Voronoi":
                radius, falloff = params["radius"], params["falloff"]
                padded_noise = generate_voronoi_noise(
                    world_size+4,
                    world_size+4,
                    scale=scale,
                    radius=radius,
                    falloff=falloff,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    seed=seed,
                )
            elif noise_type == "Perlin":
                padded_noise = generate_perlin_noise(
                    world_size+4,
                    world_size+4,
                    scale=scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    seed=seed,
                )
            else:
                raise ValueError(f"噪声类型错误: {noise_type}")
            assert padded_noise is not None
            pad_noise = apply_math_operations(padded_noise, operations)
            noise_layers.append(pad_noise)
            if raw_combined is None:
                raw_combined = pad_noise.copy()
            else:
                raw_combined += pad_noise

        assert raw_combined is not None
        assert noise_layers != []
        
        
        st.session_state.noise_layers = noise_layers
        st.session_state.raw_combined = raw_combined

        st.session_state.combined = apply_math_operations(
            st.session_state.raw_combined, get_param_state()["post_operations"]
        )


if True:
    if "param_hash" not in st.session_state:
        st.session_state.param_hash = nested_hash({})
        
    if st.session_state.param_hash != nested_hash(get_param_state()):
        st.warning("参数已改变，请重新生成")


    
    if "combined" in st.session_state:
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
                    "网格规模", 2, 1000, 30, help="格子数量以及显示时的采样点数量", key="stride_slider"
                )
                stride = max(get_param_state()["world_size"] // stride_scale, 1)

        with st.expander("图表尺寸和dpi"):
            col_w, col_h, col_dpi, col_hor = st.columns(4)
            with col_w:
                w = st.slider("宽度", 4, 48, 20, 1, key="w_slider")
            with col_h:
                h = st.slider("高度", 4, 48, 16, 1, key="h_slider")
            with col_dpi:
                dpi = st.slider("图表dpi", 10, 2000, 300, 1, key="dpi_slider")
            with col_hor:
                if st.selectbox("布局方向", ["横向", "纵向"], 1, key="hor_selectbox") == "横向":
                    cr = 120
                else:
                    cr = 210

        # 添加切片控制选项
        with st.expander("🔪 切片视图设置"):
            slice_type = st.radio("切片方向", ["X轴切片", "Y轴切片"], horizontal=True, key="slice_type_radio")
            slice_pos = st.slider(
                "切片位置",
                0,
                get_param_state()["world_size"] - 1,
                get_param_state()["world_size"] // 2,
                help="选择要查看的切片位置",
                key="slice_pos_slider",
            )

        # 代码生成控制
        with st.expander("👾 代码生成设置"):
            gen_ascii_terrian = st.checkbox("生成字符画", [True, False], key="gen_ascii_terrian_checkbox")
            ascii_terrian_chars = list(
                st.text_input("代表不同高度的字符(由低到高, 数量不限)", ".*#")
            )
            ascii_terrian_size = st.slider("字符画尺寸", 20, 300, 60, 20, key="ascii_terrian_size_slider")

        tabs = st.tabs(
            [f"Layer {i+1}" for i in range(len(st.session_state.noise_layers))]
            + ["Combined"]
            + ["参数和代码"],
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

        with st.spinner("Wait for it..."):
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
                        np.arange(get_param_state()["world_size"]),
                        slice_data,
                        color=color,
                        linestyle=linestyle,
                        label=label,
                    )
                    ax.set_title(f"{title} (X = {slice_pos})", color="white")
                else:
                    slice_data = data[:, slice_pos]
                    ax.plot(
                        np.arange(get_param_state()["world_size"]),
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
                    st.session_state.raw_combined[1:-1, 1:-1],
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
                for i, pad_layer in enumerate(st.session_state.noise_layers):
                    layer = pad_layer[1:-1, 1:-1]
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

            for i, pad_layer in enumerate(st.session_state.noise_layers):
                layer = pad_layer[1:-1, 1:-1]
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

            # 生成代码页
            with tabs[-1]:

                # 生成代码函数
                def generate_python_code():
                    # 收集基础参数
                    base_config = {
                        "post_operations": get_param_state()["post_operations"],
                        "seed": get_param_state()["seed"],
                    }

                    # 构建代码模
                    newline_char = "\n"

                    code_template = f"""
'''  world_size:{get_param_state()['world_size']}x{get_param_state()['world_size']}, chars:{ascii_terrian_chars}
{terrain_to_ascii(st.session_state.combined, chars=ascii_terrian_chars, size=(ascii_terrian_size, ascii_terrian_size)) if gen_ascii_terrian else ''}



'''
from noises.perlin import Perlin
from noises.voronoi import Voronoi 
import math
from vmath import vec2i

from noises.utils import generate_layer_noise, apply_operations_area

base_config = {pprint.pformat(base_config, indent=4).replace('newline_char', '    ')}



layers_config = {pprint.pformat(get_param_state()['layer_params'], indent=4).replace('newline_char', '    ').replace('newline_char', '    ')}




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
    return result




# ==== 可视化 ====
if __name__ == '__main__':

    from array2d import array2d
    import random
    from noises.utils import terrain_to_ascii
    
    result = noise_area(vec2i(0, 0), {get_param_state()['world_size']}, {get_param_state()['world_size']}, 1, base_config['seed'])
    
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
