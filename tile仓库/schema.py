from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, ForeignKeyConstraint, create_engine, insert, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

class Tileset(Base):
    '''
    ```
    __tablename__ = "tilesets"
    id =                Column(Integer, primary_key=True, comment="Tileset唯一标识ID")
    alias =             Column(String, comment="Tileset别名")
    is_deprecated =     Column(Boolean, default=False, comment="是否已弃用")
    comments =          Column(String, comment="Tileset备注信息")
    ```
    '''
    __tablename__ = "tilesets"
    id =                Column(Integer, primary_key=True, autoincrement=True, comment="Tileset唯一标识ID")
    alias =             Column(String, comment="Tileset别名")
    is_deprecated =     Column(Boolean, default=False, comment="是否已弃用")
    next_tile_id =      Column(Integer, default=0, comment="下一个Tile的ID")
    comments =          Column(String, comment="Tileset备注信息")

class Tile(Base):
    '''
    ```
    __tablename__ = 'tiles'
    tileset_id =        Column(Integer, primary_key=True, comment="所属Tileset的ID")
    id =                Column(Integer, primary_key=True, comment="Tile在Tileset内的唯一ID")
    char =              Column(String, comment="Tile显示的字符")
    fg =                Column(String, comment="Tile前景色")
    bg =                Column(String, comment="Tile背景色")
    alias =             Column(String, comment="Tile别名")
    is_deprecated =     Column(Boolean, default=False, comment="是否已弃用")
    comments =          Column(String, comment="Tile备注信息")
    ```
    
    '''
    __tablename__ = 'tiles'
    tileset_id =        Column(Integer, primary_key=True, comment="所属Tileset的ID")
    id =                Column(Integer, primary_key=True, comment="Tile在Tileset内的唯一ID")
    char =              Column(String, comment="Tile显示的字符")
    fg =                Column(String, comment="Tile前景色")
    bg =                Column(String, comment="Tile背景色")
    alias =             Column(String, comment="Tile别名")
    is_deprecated =     Column(Boolean, default=False, comment="是否已弃用")
    comments =          Column(String, comment="Tile备注信息")


def reset_next_tile_ids(engine):
    """重置next_tile_id为当前tileset中的最大tile id + 1, 仅在数据库初始化时, tile全部存完再执行"""
    Session = sessionmaker(bind=engine)
    with Session() as session:
        print(1)
        for tileset in session.query(Tileset).all():
            # 获取当前tileset的tile id最大值
            max_tile_id = session.query(func.max(Tile.id)).filter_by(tileset_id=tileset.id).scalar()
            # 更新next_tile_id
            tileset.next_tile_id = max_tile_id + 1
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"初始化失败: {str(e)}")
