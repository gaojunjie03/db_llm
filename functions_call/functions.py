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
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

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
        result.append({
            "table_name":table_name,
            "table_real_name":ds.SCHEMA+"."+table_real_name,
            "cols":columns
        })
    ret=json.dumps({
         "data_syntax":"请使用"+dbutil.DATABASE_DICT[ds.TYPE]+"数据库语法生成sql", 
         "result":result
    }, ensure_ascii=False)
    return ret,False
    
# 查询数据库表
@functions_decorator
def query_table(params,ds_collection_info:DsCollectionInfo):
    conn = None
    try:
        sql=params["sql"]
        logging.info("最终执行的sql语句："+sql)
        ds=dbutil.Datasource(**json.loads(ds_collection_info.datasource))
        ds.unquote()
        conn=dbutil.get_conn(ds)
        if int(ds.TYPE) not in [4,3,6]:
            sql=text(sql)
        records=pd.read_sql_query(sql,conn)
        return records.to_json(orient="records",force_ascii=False),True
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
    return f"柱状图:\n\n![bar]({bar_b64})\n\n饼图:\n\n![pie]({pie_b64})",True