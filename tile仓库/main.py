import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, select, update, insert, delete
from sqlalchemy.orm import sessionmaker
from schema import Tile, Tileset, Base  
import traceback


st.set_page_config(layout="wide")

import json
from typing import List, Dict, Any

def export_to_json(engine, indent: int = 2) -> str:
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # 准备导出的数据结构
    data: Dict[str, List[Dict[str, Any]]] = {
        "tilesets": [],
        "tiles": []
    }
    
    try:
        # 导出Tilesets数据
        tilesets = session.query(Tileset).all()
        for tileset in tilesets:
            data["tilesets"].append({
                "id": tileset.id,
                "alias": tileset.alias,
                "is_deprecated": tileset.is_deprecated,
                "comments": tileset.comments
            })
        
        # 导出Tiles数据
        tiles = session.query(Tile).all()
        for tile in tiles:
            data["tiles"].append({
                "tileset_id": tile.tileset_id,
                "id": tile.id,
                "char": tile.char,
                "fg": tile.fg,
                "bg": tile.bg,
                "alias": tile.alias,
                "is_deprecated": tile.is_deprecated,
                "comments": tile.comments
            })
        
        s = json.dumps(data, ensure_ascii=False, indent=indent)
        session.close()
        return s
        
    except Exception as e:
        print(f"导出数据时出错: {e}")
    finally:
        session.close()
    
    

def get_data_changes(edited_df, original_df, key_columns):
    """
    检测两个DataFrame之间的差异（修改、新增、删除的行），支持NaN值的准确识别
    
    参数:
        edited_df: 编辑后的DataFrame
        original_df: 原始DataFrame 
        key_columns: 用于匹配行的主键列名列表或单个列名
        
    返回:
        dict: 包含以下键的字典
            'modified': 修改过的行（保持edited_df的列顺序）
            'added': 新增的行（保持edited_df的列顺序）
            'deleted': 删除的行（保持original_df的列顺序）
    """
    # 确保key_columns是列表
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    key_columns = list(key_columns)
    
    # 合并数据并标记来源
    merged = pd.merge(
        edited_df,
        original_df,
        on=key_columns,
        how='outer',
        suffixes=('_edited', '_original'),
        indicator=True
    )
    
    # 分割不同变更类型
    added = merged[merged['_merge'] == 'left_only'].copy()
    deleted = merged[merged['_merge'] == 'right_only'].copy()
    modified_candidates = merged[merged['_merge'] == 'both'].copy()
    
    # 识别真正修改的行（处理NaN值）
    modified_mask = pd.Series(False, index=modified_candidates.index)
    for col in edited_df.columns:
        if col not in key_columns:
            ed_col, org_col = f'{col}_edited', f'{col}_original'
            # 精确比较（包含NaN处理）
            modified_mask |= modified_candidates[ed_col].ne(modified_candidates[org_col]) | (
                modified_candidates[ed_col].isna() ^ modified_candidates[org_col].isna()
            )
    
    modified = modified_candidates[modified_mask].copy()
    
    def _clean_columns(df, suffix, src_columns):
        """辅助函数：清理列名并保持源数据列顺序"""
        df = df.copy()
        for col in src_columns:
            if col not in key_columns:
                df.rename(columns={f'{col}_{suffix}': col}, inplace=True)
        return df[src_columns].drop(columns='_merge', errors='ignore')
    
    return {
        'modified': _clean_columns(modified, 'edited', edited_df.columns),
        'added': _clean_columns(added, 'edited', edited_df.columns),
        'deleted': _clean_columns(deleted, 'original', original_df.columns)
    }


# 初始化数据库连接
@st.cache_resource
def init_engine():
    
    # REPOSITORY_VERSION = "v_0.0.0.1"
    REPOSITORY_VERSION = "fake_data"
    
    engine = create_engine(f'sqlite:///{REPOSITORY_VERSION}.db')
    Base.metadata.create_all(engine)
    
    return engine

engine = init_engine()
Session = sessionmaker(bind=engine)

def tileset_editor_page():
    st.title("Tileset 编辑器")
    
    with Session() as session:
        # 1. Tileset 选择和管理
        st.header("Tileset 管理")
        
        # 创建两列布局
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Tileset 选择器
            
            # 添加新 Tileset
            with st.expander("添加新 Tileset"):
                with st.form("new_tileset_form"):
                    new_alias = st.text_input("别名")
                    new_comments = st.text_area("备注")
                    if st.form_submit_button("创建"):
                        new_tileset = Tileset(
                            alias=new_alias,
                            comments=new_comments
                        )
                        session.add(new_tileset)
                        session.commit()
                        st.success("创建成功")
                        st.rerun()
        
        with col2:
            # 显示和编辑选中的 Tileset
            tileset_df = pd.read_sql(select(Tileset), con=engine)
            
            # 可编辑的 Tileset 表单
            st.subheader("编辑 Tileset 属性")
            edited_tileset_df = st.data_editor(
                tileset_df.sort_values(by=['id'], ascending=False),
                key="tileset_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True),
                    "alias": st.column_config.TextColumn("别名"),
                    "is_deprecated": st.column_config.CheckboxColumn("已弃用"),
                    "comments": st.column_config.TextColumn("备注")
                }
            )
            
            # 保存 Tileset 修改
            if st.button("保存 Tileset 修改"):
                # Get fresh data from database for comparison
                original_tileset_df = pd.read_sql(select(Tileset), con=engine)
                
                changed = get_data_changes(
                    edited_tileset_df, 
                    original_tileset_df,
                    ['id']
                )
                print(changed)
                if not changed['modified'].empty or not changed['added'].empty or not changed['deleted'].empty:
                    try:
                        # 初始化计数器
                        modified_count = 0
                        added_count = 0
                        deleted_count = 0
                        for _, row in changed['modified'].iterrows():
                            session.execute(
                                update(Tileset)
                                .where(Tileset.id == row['id'])
                                .values(
                                    alias=row['alias'],
                                    is_deprecated=row['is_deprecated'],
                                    comments=row['comments']
                                )
                            )
                            modified_count += 1
                        for _, row in changed['added'].iterrows():
                            session.execute(
                                insert(Tileset)
                                .values(
                                    id=row["id"],
                                    alias=row['alias'],
                                    is_deprecated=row['is_deprecated'],
                                    comments=row['comments']
                                )
                            )
                            added_count += 1
                        for _, row in changed['deleted'].iterrows():
                            session.execute(
                                delete(Tileset)
                                .where(
                                    Tileset.id==row["id"]
                                )
                            )
                            deleted_count += 1
                        session.commit()
                        
                        message = "修改已保存！"
                        if modified_count > 0:
                            message += f" 修改了 {modified_count} 条记录"
                        if added_count > 0:
                            message += f" 新增了 {added_count} 条记录"
                        if deleted_count > 0:
                            message += f" 删除了 {deleted_count} 条记录"
                        st.success(message)
                    except Exception as e:
                        st.error("保存失败")
                        st.exception(e)
                else:
                    st.info("没有检测到修改")

def tile_editor_page():
    st.title("tile编辑器")
    
    with Session() as session:
        # 1. 获取所有 Tileset 供用户选择
        all_tilesets = pd.read_sql(select(Tileset), con=engine)
        
        # 创建多选框让用户选择要查看的 Tileset
        selected_options = st.multiselect(
            "选择要查看的 Tileset",
            options=all_tilesets['id'],
            format_func=lambda x: f"ID:{x} - {all_tilesets.loc[all_tilesets['id'] == x, 'alias'].values[0]}"
        )
        if all_tilesets['id'].shape[0] == 0:
            st.warning("请新建一个 Tileset")
            return
        elif not selected_options:
            st.warning("请至少选择一个 Tileset")
            return
        
        query = select(Tile).where(Tile.tileset_id.in_(selected_options))
        tiles_df = pd.read_sql(query, con=engine)
        
        # 2. 显示可编辑表格
        st.subheader("编辑 Tile 数据")
        edited_df = st.data_editor(
            tiles_df.sort_values(by=['tileset_id', 'id'], ascending=False),
            key="tile_editor",
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "tileset_id": st.column_config.SelectboxColumn("tileset_id", options=selected_options, required=True),
                "id": st.column_config.NumberColumn("id", required=True),
                "char": st.column_config.TextColumn("char"),
                "fg": st.column_config.TextColumn("fg"),
                "bg": st.column_config.TextColumn("bg"),
                "alias": st.column_config.TextColumn("alias"),
                "is_deprecated": st.column_config.CheckboxColumn("is_deprecated", default=False),
                "comments": st.column_config.TextColumn("comments")
            }
        )
        
        # 3. 比较并提交修改
        if st.button("保存修改"):
            # Get fresh data from database for comparison
            original_df = pd.read_sql(query, con=engine)
            
            # 找出被修改的行
            changed = get_data_changes(edited_df, original_df, ['tileset_id', 'id'])
            print(changed)
            if not changed['modified'].empty or not changed['added'].empty or not changed['deleted'].empty:
                try:
                    # 初始化计数器
                    modified_count = 0
                    added_count = 0
                    deleted_count = 0
                    
                    # 更新数据库 - 修改记录
                    for idx, row in changed['modified'].iterrows():
                        # 获取原始ID
                        tileset_id = row['tileset_id']
                        tile_id = row['id']
                        
                        # 更新记录
                        session.execute(
                            update(Tile)
                            .where(Tile.tileset_id == tileset_id)
                            .where(Tile.id == tile_id)
                            .values(
                                char=row['char'],
                                fg=row['fg'],
                                bg=row['bg'],
                                alias=row['alias'],
                                is_deprecated=row['is_deprecated'],
                                comments=row['comments']
                            )
                        )
                        modified_count += 1
                    
                    # 添加新记录
                    for _, row in changed['added'].iterrows():
                        session.execute(
                            insert(Tile)
                            .values(
                                id=row["id"],
                                tileset_id=row["tileset_id"],
                                char=row['char'],
                                fg=row['fg'],
                                bg=row['bg'],
                                alias=row['alias'],
                                is_deprecated=row['is_deprecated'],
                                comments=row['comments']
                            )
                        )
                        added_count += 1
                    
                    # 删除记录
                    for _, row in changed['deleted'].iterrows():
                        session.execute(
                            delete(Tile)
                            .where(Tile.tileset_id == row["tileset_id"])
                            .where(Tile.id == row["id"])
                        )
                        deleted_count += 1
                    
                    session.commit()
                    
                    # 构建成功消息
                    message = "修改已保存！"
                    if modified_count > 0:
                        message += f" 修改了 {modified_count} 条记录"
                    if added_count > 0:
                        message += f" 新增了 {added_count} 条记录"
                    if deleted_count > 0:
                        message += f" 删除了 {deleted_count} 条记录"
                    
                    st.success(message)
                    
                except Exception as e:
                    session.rollback()
                    st.error("保存失败")
                    st.exception(e)
            else:
                st.info("没有检测到修改")

def main():
    tab1, tab2, tab3 = st.tabs(["tile编辑器", "tilset编辑器", "导出"])
    with tab1:
        tile_editor_page()
    with tab2:
        tileset_editor_page()
    with tab3:
        st.code(export_to_json(engine), 'json')

if __name__ == "__main__":
    main()