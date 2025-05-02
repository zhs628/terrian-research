from vmath import vec3i

class AsciiSprite:
    def __init__(self, char: str, fg: str | None = None, bg: str | None = None):
        self.char = char
        self.fg = self.hex_color_to_vec3i(fg) if fg else None
        self.bg = self.hex_color_to_vec3i(bg) if bg else None

    @staticmethod
    def hex_color_to_vec3i(color: str) -> vec3i:
        return vec3i(
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
        )

class Tileset:
    def __init__(self, id: int, sprites: list[AsciiSprite]):
        self.id = id
        self.sprites = sprites


class TilesetId:
    Chasm = 0
    Water = 1
    Ground = 2
    Floor = 3
    Slime = 4
    Grass = 5
    Fire = 6
    Wall = 7
    Tree = 8
    Special = 9
    
    Direction = 10
    Height = 11


_db = [
    # 裂隙
    Tileset(TilesetId.Chasm, [
        AsciiSprite('　'),
    ]),
    # 水
    Tileset(TilesetId.Water, [
        AsciiSprite('　', bg="#00aaff"),
    ]),
    # 地基
    Tileset(TilesetId.Ground, [
        AsciiSprite('・'),
        AsciiSprite('・', bg="#314c6e"),  # 高海拔
        AsciiSprite('・', bg="#ffda79"),  # 低海拔
        AsciiSprite('・', bg="#f9ca24"), # 中海拔
    ]),
    # 地板
    Tileset(TilesetId.Floor, [
        AsciiSprite('＊'),
    ]),
    # 粘液
    Tileset(TilesetId.Slime, [
        AsciiSprite('～', bg="#ff44ef"),
    ]),
    # 草
    Tileset(TilesetId.Grass, [
        AsciiSprite('🌿'),
    ]),
    # 火焰
    Tileset(TilesetId.Fire, [
        AsciiSprite('🔥', bg="#ff6229"),
    ]),
    # 墙体
    # https://unicode.party/?query=mountain
    Tileset(TilesetId.Wall, [
        AsciiSprite('🧱'),
        AsciiSprite('🪨 '),
        AsciiSprite('🧊'),
        AsciiSprite('⛰️'),
        AsciiSprite('🏔️'),
        AsciiSprite('🗻'),
        AsciiSprite('🌋'),
    ]),
    # 树木
    # https://unicode.party/?query=tree
    Tileset(TilesetId.Tree, [
        AsciiSprite('🌲'),
        AsciiSprite('🎄'),
        AsciiSprite('🌴', bg="#ffda79"),
        AsciiSprite('🌴', bg="#f9ca24"),
        AsciiSprite('🎋'),
        AsciiSprite('🌳'),
    ]),
    
    # 特殊点
    Tileset(TilesetId.Special, [
        AsciiSprite('🛕'),
    ]),
    
    Tileset(TilesetId.Direction, [
            AsciiSprite('↑​'),
            AsciiSprite('↗'),
            AsciiSprite('→'),
            AsciiSprite('↘'),
            AsciiSprite('↓'),
            AsciiSprite('↙️​​'),
            AsciiSprite('←'),
            AsciiSprite('↖️'),
        ]),
    
    Tileset(TilesetId.Height, [
            AsciiSprite('１', bg="#fef4f4", fg="#44bd32"),
            AsciiSprite('２', bg="#fdeff2", fg="#44bd32"),
            AsciiSprite('３', bg="#e9dfe5", fg="#44bd32"),
            AsciiSprite('４', bg="#e4d2d8", fg="#44bd32"),
            AsciiSprite('５', bg="#f6bfbc", fg="#44bd32"),
            AsciiSprite('６', bg="#f5b1aa", fg="#44bd32"),
            AsciiSprite('７', bg="#f5b199", fg="#44bd32"),
            AsciiSprite('８', bg="#efab93", fg="#44bd32"),
        ]),
]

def get_sprite(tileset_id: int, index: int) -> AsciiSprite:
    s = _db[tileset_id].sprites[index]
    assert '​​' not in s.char  # 避免隐形空格
    return s
