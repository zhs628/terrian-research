from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, ForeignKeyConstraint, create_engine
from sqlalchemy.orm import declarative_base

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
    id =                Column(Integer, primary_key=True, comment="Tileset唯一标识ID")
    alias =             Column(String, comment="Tileset别名")
    is_deprecated =     Column(Boolean, default=False, comment="是否已弃用")
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


