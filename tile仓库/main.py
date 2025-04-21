import dis
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, select, update, insert, delete
from sqlalchemy.orm import sessionmaker
from schema import Tile, Tileset, Base
import traceback
from datetime import datetime
import unicodedata


st.set_page_config(layout="wide")

import json

# REPOSITORY_VERSION = "v_0.0.0.1"
REPOSITORY_VERSION = "fake_data"


def export_to_json(engine, indent: int = 2) -> str:

    Session = sessionmaker(bind=engine)
    session = Session()

    # 准备导出的数据结构
    data = {
        "version": str(REPOSITORY_VERSION),
        "created_at": str(datetime.now()),
        "tilesets": [],
    }

    try:
        # 导出Tilesets数据
        tilesets = session.query(Tileset).all()

        tileset_id_to_data_index = {}
        for i, tileset in enumerate(tilesets):
            data["tilesets"].append(
                {"id": tileset.id, "name": tileset.name, "tiles": []}
            )
            tileset_id_to_data_index[tileset.id] = i

        # 导出Tiles数据
        tiles = session.query(Tile).all()
        for tile in tiles:
            data["tilesets"][tileset_id_to_data_index[tile.tileset_id]]["tiles"].append(
                {
                    "id": tile.id,
                    "char": tile.char,
                    "fg": tile.fg,
                    "bg": tile.bg,
                }
            )

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
        how="outer",
        suffixes=("_edited", "_original"),
        indicator=True,
    )

    # 分割不同变更类型
    added = merged[merged["_merge"] == "left_only"].copy()
    deleted = merged[merged["_merge"] == "right_only"].copy()
    modified_candidates = merged[merged["_merge"] == "both"].copy()

    # 识别真正修改的行（处理NaN值）
    modified_mask = pd.Series(False, index=modified_candidates.index)
    for col in edited_df.columns:
        if col not in key_columns:
            ed_col, org_col = f"{col}_edited", f"{col}_original"
            # 精确比较（包含NaN处理）
            modified_mask |= modified_candidates[ed_col].ne(
                modified_candidates[org_col]
            ) | (
                modified_candidates[ed_col].isna() ^ modified_candidates[org_col].isna()
            )

    modified = modified_candidates[modified_mask].copy()

    def _clean_columns(df, suffix, src_columns):
        """辅助函数：清理列名并保持源数据列顺序"""
        df = df.copy()
        for col in src_columns:
            if col not in key_columns:
                df.rename(columns={f"{col}_{suffix}": col}, inplace=True)
        return df[src_columns].drop(columns="_merge", errors="ignore")

    return {
        "modified": _clean_columns(modified, "edited", edited_df.columns),
        "added": _clean_columns(added, "edited", edited_df.columns),
        "deleted": _clean_columns(deleted, "original", original_df.columns),
    }


def update_tileset(tileset: Tileset) -> bool:

    with Session() as session:
        try:
            session.execute(
                update(Tileset)
                .where(Tileset.id == getattr(tileset, "id"))
                .values(
                    name=getattr(tileset, "name"), comments=getattr(tileset, "comments")
                )
            )
            session.commit()
            st.toast(f"修改成功 {tileset}", icon="✅")
            return True
        except Exception as e:
            session.rollback()
            st.error("失败")
            st.exception(e)
            return False


def add_tileset(tileset: Tileset) -> bool:
    with Session() as session:
        try:
            session.execute(
                insert(Tileset).values(
                    name=getattr(tileset, "name"), comments=getattr(tileset, "comments")
                )
            )
            session.commit()
            st.toast(f"添加成功 {tileset}", icon="✅")
            return True
        except Exception as e:
            session.rollback()
            st.error("失败")
            st.exception(e)
            return False


def delete_tileset(tileset: Tileset) -> bool:
    with Session() as session:
        try:
            session.execute(delete(Tileset).where(Tileset.id == getattr(tileset, "id")))

            session.execute(
                delete(Tile).where(Tile.tileset_id == getattr(tileset, "id"))
            )

            session.commit()
            st.toast(f"删除成功 {tileset} ", icon="✅")
            return True
        except Exception as e:
            session.rollback()
            st.error("失败")
            st.exception(e)
            return False


def tileset_editor_page():
    st.title("tileset编辑器")
    all_tilesets = pd.read_sql(select(Tileset), con=engine)

    for i, tileset in enumerate(all_tilesets.itertuples(index=True)):
        with st.expander(
            f"{i}. {getattr(tileset, 'id')} | {getattr(tileset, 'name')}",
            expanded=st.session_state.get(
                f"{i}.tileset_expander_last_status", default=False
            ),
        ):
            index = getattr(tileset, "Index")

            # 闭包工厂
            def make_update_action(field, i, tileset, index):
                def _update_action():

                    # 修改对应expander状态
                    for ii in range(all_tilesets.shape[0]):
                        st.session_state[f"{ii}.tileset_expander_last_status"] = False
                    st.session_state[f"{i}.tileset_expander_last_status"] = True

                    # 干正事
                    new_value = st.session_state[f"{i}.tileset_update_{field}_new_val"]
                    new_tileset = tileset._replace(**{field: new_value})
                    if update_tileset(new_tileset):
                        all_tilesets.loc[index, field] = new_value

                return _update_action

            def make_delete_action(i, tileset, index):

                def _delete_action():
                    if delete_tileset(tileset):
                        all_tilesets.drop(index, inplace=True)
                        st.toast("✅ 删除成功")
                    else:
                        st.toast("❌ 删除失败")

                return _delete_action

            st_columns = st.columns(4)
            # ------------ 编辑 Tileset
            with st_columns[0]:
                if f"{i}.tileset_update_id_new_val" not in st.session_state:
                    st.session_state[f"{i}.tileset_update_id_new_val"] = tileset.id
                _update_id_action = make_update_action("id", i, tileset, index)
                new_val = st.number_input(
                    f"{i}.  id",
                    step=1,
                    on_change=_update_id_action,
                    key=f"{i}.tileset_update_id_new_val",
                    disabled=True,
                )

            with st_columns[1]:
                _update_name_action = make_update_action("name", i, tileset, index)
                new_val = st.text_input(
                    f"{i}.  name",
                    value=tileset.name,
                    on_change=_update_name_action,
                    key=f"{i}.tileset_update_name_new_val",
                )

            with st_columns[2]:
                _update_comments_action = make_update_action(
                    "comments", i, tileset, index
                )
                new_val = st.text_area(
                    f"{i}.  comments",
                    value=tileset.comments,
                    on_change=_update_comments_action,
                    key=f"{i}.tileset_update_comments_new_val",
                )

            # ------------ 删除 Tileset
            with st_columns[3]:


                corr_content = f"id:{getattr(tileset, 'id')},name:{getattr(tileset, 'name')},tiles:{Session().query(Tile).filter_by(tileset_id = getattr(tileset, 'id')).count()}"
                content = st.text_area("请输入: " + corr_content, value="")
                if content == corr_content:
                    st.button(
                        f"{i}. 删除", on_click=make_delete_action(i, tileset, index)
                    )

    # ------------ 新增 Tileset
    def _add_action():
        # 创建一个新的 Tileset 实例
        new_tileset = Tileset(name="", comments="")

        # 尝试添加新的 Tileset 到数据库
        if add_tileset(new_tileset):
            # 获取新添加的 Tileset 的 ID
            new_tileset_id = new_tileset.id

            # 将新 Tileset 添加到 all_tilesets DataFrame 中
            all_tilesets.loc[len(all_tilesets)] = [
                new_tileset_id,
                new_tileset.name,
                new_tileset.next_tile_id,
                new_tileset.comments,
            ]

    st.button(f"Add TileSet", on_click=_add_action)


def tile_editor_page():
    st.title("Tile 编辑器")
    all_tilesets = pd.read_sql(select(Tileset), con=engine)

    # 创建多选框让用户选择要查看的 Tileset
    selected_options = st.multiselect(
        "选择要查看的 Tileset",
        options=all_tilesets["id"],
        format_func=lambda x: f"ID:{x} - {all_tilesets.loc[all_tilesets['id'] == x, 'name'].values[0]}",
    )
    if all_tilesets["id"].shape[0] == 0:
        st.warning("请新建一个 Tileset")
        return
    elif not selected_options:
        st.warning("请至少选择一个 Tileset")
        return

    query = select(Tile).where(Tile.tileset_id.in_(selected_options))
    all_tiles = pd.read_sql(query, con=engine)
    print(all_tiles)

    for i, tile in enumerate(all_tiles.itertuples(index=True)):
        with st.expander(
            f"{i}.  {getattr(tile, 'tileset_id')} | {getattr(tile, 'id')} | {getattr(tile, 'char')} | {unicodedata.name(getattr(tile, 'char'))}",
            expanded=st.session_state.get(
                f"{i}.tile_expander_last_status", default=False
            ),
        ):
            index = getattr(tile, "Index")

            # 闭包工厂
            def make_update_action(field, i, tile, index):
                def _update_action():
                    # 修改对应expander状态
                    for ii in range(all_tiles.shape[0]):
                        st.session_state[f"{ii}.tile_expander_last_status"] = False
                    st.session_state[f"{i}.tile_expander_last_status"] = True

                    # 干正事
                    new_value = st.session_state[f"{i}.tile_update_{field}_new_val"]
                    new_tile = tile._replace(**{field: new_value})
                    if update_tile(new_tile):
                        all_tiles.loc[index, field] = new_value

                return _update_action

            def make_delete_action(i, tile, index):
                def _delete_action():
                    # 删除对应的tile
                    if delete_tile(tile):  # 假设有一个delete_tile函数来执行实际删除操作
                        all_tiles.drop(index=index, inplace=True)
                        # 重置所有expander状态
                        for ii in range(all_tiles.shape[0]):
                            st.session_state[f"{ii}.tile_expander_last_status"] = False
                        st.toast("✅ 删除成功")
                    else:
                        st.toast("❌ 删除失败")

                return _delete_action

            st_columns = st.columns(5)
            # ------------ 编辑 Tile
            with st_columns[0]:
                if f"{i}.tile_update_tileset_id_new_val" not in st.session_state:
                    st.session_state[f"{i}.tile_update_tileset_id_new_val"] = (
                        tile.tileset_id
                    )
                _update_tileset_id_action = make_update_action(
                    "tileset_id", i, tile, index
                )
                new_val = st.selectbox(
                    f"{i}.  tileset_id",
                    options=all_tilesets["id"],
                    on_change=_update_tileset_id_action,
                    key=f"{i}.tile_update_tileset_id_new_val",
                    disabled=True,
                )

            with st_columns[0]:
                if f"{i}.tile_update_id_new_val" not in st.session_state:
                    st.session_state[f"{i}.tile_update_id_new_val"] = tile.id
                _update_id_action = make_update_action("id", i, tile, index)
                new_val = st.number_input(
                    f"{i}.  id",
                    step=1,
                    on_change=_update_id_action,
                    key=f"{i}.tile_update_id_new_val",
                    disabled=True,
                )

            with st_columns[1]:
                _update_char_action = make_update_action("char", i, tile, index)
                new_val = st.text_input(
                    f"{i}.  char",
                    value=tile.char,
                    on_change=_update_char_action,
                    key=f"{i}.tile_update_char_new_val",
                )

            with st_columns[2]:
                _update_fg_action = make_update_action("fg", i, tile, index)
                new_val = st.color_picker(
                    f"{i}.  fg",
                    value=tile.fg,
                    on_change=_update_fg_action,
                    key=f"{i}.tile_update_fg_new_val",
                )

            with st_columns[2]:
                _update_bg_action = make_update_action("bg", i, tile, index)
                new_val = st.color_picker(
                    f"{i}.  bg",
                    value=tile.bg,
                    on_change=_update_bg_action,
                    key=f"{i}.tile_update_bg_new_val",
                )

            with st_columns[3]:
                _update_comments_action = make_update_action("comments", i, tile, index)
                new_val = st.text_area(
                    f"{i}.  comments",
                    value=tile.comments,
                    on_change=_update_comments_action,
                    key=f"{i}.tile_update_comments_new_val",
                )

            # ------------ 删除 Tile
            with st_columns[4]:

                corr_content = f"id:{getattr(tile, 'id')},unicodedata_name:{unicodedata.name(getattr(tile, 'char'))}"
                content = st.text_area("请输入: " + corr_content, value="")
                if content == corr_content:
                    st.button(f"{i}. 删除", on_click=make_delete_action(i, tile, index))

    add_col1, add_col2 = st.columns(2)

    with add_col1:
        new_tile_tileset_id = st.selectbox(
            f"新增tile至:",
            options=all_tilesets["id"],
        )

    # ------------ 新增 Tile
    def _add_action():
        # 创建一个新的 Tile 实例
        new_tile = Tile(tileset_id=new_tile_tileset_id, name="", comments="")

        # 尝试添加新的 Tile 到数据库
        if add_tile(new_tile):
            # 获取新添加的 Tile 的 ID
            new_tile_id = new_tile.id

            # 将新 Tile 添加到 all_tiles DataFrame 中
            all_tiles.loc[len(all_tiles)] = [
                new_tile.tileset_id,
                new_tile.id,
                new_tile.char,
                new_tile.fg,
                new_tile.bg,
                new_tile.comments,
            ]

    with add_col2:
        st.button(f"Add Tile", on_click=_add_action)


# 更新和删除 Tile 的函数
def update_tile(tile: Tile) -> bool:
    with Session() as session:
        try:
            session.execute(
                update(Tile)
                .where(Tile.tileset_id == getattr(tile, "tileset_id"))
                .where(Tile.id == getattr(tile, "id"))
                .values(
                    char=getattr(tile, "char"),
                    fg=getattr(tile, "fg"),
                    bg=getattr(tile, "bg"),
                    comments=getattr(tile, "comments"),
                )
            )
            session.commit()
            st.toast(f"修改成功 {tile}", icon="✅")
            return True
        except Exception as e:
            session.rollback()
            st.error("失败")
            st.exception(e)
            return False


def add_tile(tile: Tile) -> bool:

    if tile.id is not None:
        raise Exception("Tile id should be None")

    with Session() as session:
        try:

            next_tile_id = (
                session.query(Tileset)
                .filter_by(id=getattr(tile, "tileset_id"))
                .first()
                .next_tile_id
            )

            # 插入新的 Tile
            session.execute(
                insert(Tile).values(
                    tileset_id=getattr(tile, "tileset_id"),
                    id=next_tile_id,
                    char=getattr(tile, "char"),
                    fg=getattr(tile, "fg"),
                    bg=getattr(tile, "bg"),
                    comments=getattr(tile, "comments"),
                )
            )

            # 更新所属 tileset 的 next_tile_id
            session.query(Tileset).filter_by(id=1).update(
                {"next_tile_id": next_tile_id + 1}
            )

            session.commit()
            st.toast(f"添加成功 {tile}", icon="✅")
            return True

        except Exception as e:
            session.rollback()
            st.error("失败")
            st.exception(e)
            return False


def delete_tile(tile: Tile) -> bool:
    with Session() as session:
        try:
            session.execute(
                delete(Tile).where(
                    Tile.tileset_id == getattr(tile, "tileset_id"),
                    Tile.id == getattr(tile, "id"),
                )
            )
            session.commit()
            st.toast(f"删除成功 {tile}", icon="✅")
            return True
        except Exception as e:
            session.rollback()
            st.error("失败")
            st.error(traceback.format_exc())
            return False


# 初始化数据库连接
@st.cache_resource
def init_engine():

    engine = create_engine(f"sqlite:///{REPOSITORY_VERSION}.db")
    Base.metadata.create_all(engine)
    return engine


engine = init_engine()
Session = sessionmaker(bind=engine)


def main():
    tab1, tab2, tab3 = st.tabs(["tile编辑器", "tilset编辑器", "导出"])
    with tab1:
        tile_editor_page()
    with tab2:
        tileset_editor_page()
    with tab3:
        st.code(export_to_json(engine), "json")


if __name__ == "__main__":

    main()
