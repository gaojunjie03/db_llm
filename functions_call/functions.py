import pandas as pd
from sqlalchemy import text
import db.dbutil as dbutil
import json
import constants
from  entity.ds_collection_info import DsCollectionInfo
from db.db_tools import open_milvus
from log.log import logging
from functools import wraps
import io
import base64
import matplotlib.pyplot as plt
from entity.result import Result
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
from openai_harmony import DeveloperContent,ToolDescription

harmony_functions=(
                DeveloperContent.new()
                .with_instructions("")
                .with_function_tools(
                    [
                        ToolDescription.new(
                            "map_tables_fields",
                            "根据分析用户问题得到的各表中文描述获取各表对应的真实数据库表名和数据库表字段信息",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "table_name":{
                                                   "type":"string",
                                                   "description": "中文描述的数据表名",
                                                }
                                            },
                                             "required": ["table_name"],
                                        },
                                        "description": " 需要映射的表名列表，使用中文描述",
                                    }
                                },
                                "required": ["parameters"],
                            },
                        ),
                        ToolDescription.new(
                            "query_table",
                            "执行sql语句",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "sql":{
                                                   "type":"string",
                                                   "description": "sql语句",
                                                }
                                            },
                                            "required": ["sql"]
                                        },
                                        "description": "参数对象",
                                    },
                                    "required": ["parameters"],
                                },
                        ),
                        ToolDescription.new(
                            "plot_charts",
                            "根据最新的查询结果生成统计图表",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "summary":{
                                                  "type":"string",
                                                  "description":"对图表数据简单总结一两句话"
                                            },
                                            "title":{
                                                   "type":"string",
                                                   "description": "统计图表的标题",
                                            },
                                            "bar_xlabel":{
                                                   "type":"string",
                                                   "description": "横坐标名称",
                                            },
                                            "bar_ylabel":{
                                                   "type":"string",
                                                   "description": "纵坐标名称",
                                            },
                                            "label":{
                                                  "type":"array",
                                                  "description": "标签数据",
                                                   "items": {
                                                        "type": "string",
                                                    },
                                            },
                                            "data":{
                                                  "type":"array",
                                                  "description": "各标签对应的统计数据",
                                                   "items": {
                                                        "type": "number",
                                                    },
                                            },
                                            "required": ["summary","title","label","data","bar_xlabel","bar_ylabel"]
                                        },
                                        "description": "参数对象",
                                    },
                                    "required": ["parameters"],
                                },
                            },
                        ),
                    ]
                )
            )


class FunctionsException(BaseException):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]
    
def functions_decorator(func):
    @wraps(func)  
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except BaseException as e:
            raise FunctionsException(str(e))
        return result
    return wrapper


# functioncall 函数定义
# 从索引库检索表和字段真实索引
@functions_decorator
def map_tables_fields(params,ds_collection_info:DsCollectionInfo):
    result=[]
    table_names = [ table_col["table_name"] if table_col !="" else None for table_col in params]
    table_names=list(filter(lambda x: x is not None, table_names))
    ds=dbutil.Datasource(**json.loads(ds_collection_info.datasource))
    with open_milvus() as db:
        res = db.client.search(
            collection_name=ds_collection_info.collection_name,
            data=constants.sentence_model.predict(table_names),
            anns_field="table_embedding",
            limit=3,
            output_fields=["table_real_name","columns"],
        )
    for idx, hits in enumerate(res):
        hit=hits[0]
        table_name=table_names[idx]
        table_real_name=hit["entity"].get("table_real_name")
        columns=hit["entity"].get("columns")
        col_obj=json.loads(columns)
        logging.info(col_obj)
        for col in col_obj:
            col["col_real_name"]=add_quote(col["col_real_name"],ds)
        table_real_name=ds.SCHEMA+"."+add_quote(table_real_name,ds)
        need_append=True
        for rr in result:
            if rr["table_real_name"] == table_real_name:
                rr["table_name"] += ","+table_name
                need_append=False
        if need_append:
            result.append({
                    "table_name":table_name,
                    "table_real_name":table_real_name,
                    "cols":json.dumps(col_obj, ensure_ascii=False)
            })
    msg = f"""使用{dbutil.DATABASE_DICT[ds.TYPE]}数据库语法生成sql，然后调用functions.query_table"""
    return Result(msg,data=result)

def add_quote(name,ds):
    return "\""+name+"\"" if ds.TYPE==11 else name


# 查询数据库表
@functions_decorator
def query_table(params,ds_collection_info:DsCollectionInfo):
    conn = None
    try:
        sql=params["sql"]

        ds=dbutil.Datasource(**json.loads(ds_collection_info.datasource))
        ds.unquote()
        conn=dbutil.get_conn(ds)
        if int(ds.TYPE) not in [4,3,6]:
            if int(ds.TYPE) == 0:
                sql=sql.rstrip()
                if sql.endswith(";"):
                    sql=sql[:-1]
            logging.info("最终执行的sql语句："+sql)
            sql=text(sql)
        else:
            logging.info("最终执行的sql语句："+sql)
        records=pd.read_sql_query(sql,conn)
        # if len(records) == 0:
        #     return Result("""
        #     当前查询sql没有数据，请直接在final通道输出"无数据"
        #     """)
        return Result("""
            请直接在final通道输出"以下是查询结果(如果需要生成统计图表可以对我说"生成统计图表")："，同时将data的数据信息转成markdown的表格格式输出，并根据data数据简单做一两句总结。
        """,data=records.to_dict(orient="records"))
    finally:
        if conn is not None:
            conn.close()

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:image/png;base64,{b64}"

#生成图表
@functions_decorator
def plot_charts(params,ds_collection_info:DsCollectionInfo=None):
    summary=params["summary"]
    title=params["title"]
    bar_xlabel=params["bar_xlabel"]
    bar_ylabel=params["bar_ylabel"]
    data=params["data"]
    label=params["label"]
    # --- 柱状图 ---
    bar_fig, bar_ax = plt.subplots()
    bar_ax.bar(label, data, color="skyblue")
    bar_ax.set_title(title+"-柱状图")
    bar_ax.set_xlabel(bar_xlabel)
    bar_ax.set_ylabel(bar_ylabel)
    bar_b64 = fig_to_base64(bar_fig)

    # --- 饼图 ---
    pie_fig, pie_ax = plt.subplots()
    pie_ax.pie(data, labels=label, autopct='%1.1f%%')
    pie_ax.set_title(title+"-饼图")
    pie_b64 = fig_to_base64(pie_fig)
    return Result(summary,data=f"以下是统计图表展示: \n ![bar]({bar_b64}) ![pie]({pie_b64})",last_stage=True)