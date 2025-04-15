from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from faker import Faker
import random
import string
from schema import *

tiles = "🌲🌳🌴🌵🌱🌿☘️🍀🎋🎍🍂🍁🌾🌷🌸🌹🌺🌻🌼💐🌵🪴🪵🪨🏔️⛰️🌋🏕️🏜️🏝️🏖️🛖🏠🏡🏚️🏗️🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰💒⛪🕌🕍🛕🕋⛩️🗿🏟️🎪🎭🛶🚤⛵🛳️🚢🛸🚀🛰️💺🛎️🧳🌁🌃🌄🌅🌆🌇🌉🎠🎡🎢🎪🏎️🚂🚃🚄🚅🚆🚇🚈🚉🚊🚝🚞🚋🚌🚍🚎🚐🚑🚒🚓🚔🚕🚖🚗🚘🚙🛻🚚🚛🚜🏎️🏍️🛵🚲🛴🦽🦼🛺🚏🚨🚥🚦🚧⚓⛽🛞🚡🚠🚟🚁🛩️✈️🛫🛬💺🛰️🛎️🧳"

# 初始化工具
fake = Faker()
engine = create_engine('sqlite:///fake_data.db')
Session = sessionmaker(bind=engine)
session = Session()

# 创建表（如果不存在）
Base.metadata.create_all(engine)

def generate_random_tilesets(num=5):
    """生成随机Tileset数据"""
    for _ in range(num):
        yield Tileset(
            alias=fake.unique.word().capitalize() + " Set",
            is_deprecated=random.choices([True, False], weights=[0.2, 0.8])[0],
            comments=fake.sentence() if random.random() < 0.7 else None
        )

def generate_random_tiles(tileset, num=20):
    """为单个Tileset生成随机Tile数据"""
    for tile_id in range(1, num+1):
        yield Tile(
            tileset_id=tileset.id,
            id=tile_id,
            char=random.choice(tiles),
            fg=fake.hex_color(),
            bg=fake.hex_color(),
            alias=fake.word().title(),
            is_deprecated=random.choices([True, False], weights=[0.1, 0.9])[0],
            comments=fake.sentence() if random.random() < 0.6 else None
        )

# 生成并插入数据
try:
    # 生成3个tileset
    for tileset in generate_random_tilesets(3):
        session.add(tileset)
    
    session.flush()  # 生成ID但不提交
    
    # 为每个tileset生成10个tile
    for tileset in session.query(Tileset).all():
        for tile in generate_random_tiles(tileset, 10):
            session.add(tile)
    
    session.commit()
    print("数据插入成功！")
except Exception as e:
    session.rollback()
    print(f"插入数据时出错: {str(e)}")
finally:
    session.close()