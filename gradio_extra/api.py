import gradio as gr
import pandas as pd
import db.dbutil as dbutil
import json
import traceback
from sqlalchemy import text
from db.db_tools import open_sqlite,open_milvus,create_table_knowledge_collection
from log.log import logging
from entity.ds_collection_info import DsCollectionInfo
import uuid
import constants
def show_columns(all_fields,all_tables, evt: gr.SelectData):
    index=evt.index[0]
    table_name=all_tables[index]
    fields=all_fields[index]
    df_inner = pd.DataFrame(json.loads(fields))
    df_inner = df_inner.rename(columns={
          "col_real_name": "字段名",
          "col_type": "类型",
          "col_memo": "备注"
    })
    return gr.update(value=df_inner,label=table_name+"表的元数据结构信息"),index
            
def update_columns(all_fields,columns_df_data,current_table_index):
    all_fields[current_table_index]=json.dumps(columns_df_data.to_dict(orient="records"), ensure_ascii=False)
    return columns_df_data,all_fields

def insert_into_milvus(milvus_collection_name,milvus_collection_description,ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password,table_infos,field_infos):
    if milvus_collection_name is None or milvus_collection_name=="":
        return gr.update(value="❌ 请输入入库表名！", visible=True)
    if milvus_collection_description is None or milvus_collection_description=="":
        return gr.update(value="❌ 请输入描述用途！", visible=True)
    table_column_description_df=table_infos.rename(columns={"数据库表名":"table_real_name","描述":"description"})
    table_column_description_df["columns"]=field_infos
    logging.info(table_column_description_df.to_json(orient="records", force_ascii=False))
    batch_size = 20
    try:
        with open_milvus() as db:
            create_table_knowledge_collection(milvus_collection_name,db.client)
            for start in range(0, len(table_column_description_df), batch_size):
                batch = table_column_description_df.iloc[start:start+batch_size]
                embeddings = constants.sentence_model.predict(batch["description"].tolist())
                data = [
                    {"table_real_name": table_real_name, "table_embedding": embedding, "columns": columns}
                    for table_real_name, embedding, columns in zip(
                        batch["table_real_name"].to_numpy(), embeddings, batch["columns"].to_numpy()
                    )
                ]
                db.client.insert(collection_name=milvus_collection_name, data=data)
                
        with open_sqlite(need_commit=True) as db:
            ds_collection_info = DsCollectionInfo(collection_name=milvus_collection_name, description=milvus_collection_description,
                                                  datasource=json.dumps(dbutil.Datasource(database_name,database_ip, database_port, database_username, database_password, ds_type,SCHEMA=database_schema).to_dict()))
            db.session.add_all([ds_collection_info])
        return gr.update(value="✅ 入库成功！", visible=True)
    except Exception as e:
        logging.error(traceback.format_exc())
        return gr.update(value="❌ 入库失败！", visible=True)
        
    
            
def delete_tables(current_table_index,tables_df,all_tables,all_fields,columns_df):
    if current_table_index is not None and len(tables_df) < len(all_tables):
       return gr.update(value=all_tables.pop(current_table_index)),gr.update(value=all_fields.pop(current_table_index)),gr.update(value=[],label="对应表元数据信息")
    else:
       return all_tables,all_fields,columns_df
                
                
    
def get_db_metadata(ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password):  
     ds = dbutil.Datasource(database_name,database_ip, database_port, database_username, database_password, ds_type,SCHEMA=database_schema)
     df = dbutil.get_tables_field(ds)
     df_outer = df[["table_real_name", "description", "columns"]].copy()
     df_outer = df_outer.rename(columns={"table_real_name":"数据库表名","description":"描述"})
     return df_outer[["数据库表名","描述"]], df_outer["columns"].tolist(),df_outer["数据库表名"].tolist(),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)


def get_ds_collection_infos():
    with open_sqlite() as db:
        rows=db.session.query(DsCollectionInfo).all()
    return gr.update(choices=[(r.description,r.id) for r in rows]),rows


def chat_with_model(messages,history,ds_collection_info_id,all_ds_collection_info,max_token,temperature,chat_history_id,retry_num):
    if ds_collection_info_id is None:
        yield "".join(["请","选","择","查","询","的","数","据","库","信","息"]),chat_history_id
        return 
    result_ds_collection_info = list(filter(lambda x: x.id == ds_collection_info_id, all_ds_collection_info))[0]
    if len(history)==0:
        logging.info("新的聊天，需要产生新的chat_history_id")
        if chat_history_id is not None:
            logging.info("清空{}的聊天记录".format(chat_history_id))
            constants.text_generation_model.clear_chat_history(chat_history_id)
        chat_history_id=str(uuid.uuid1())
    logging.info("用户的消息:{},聊天记录id:{},数据库信息:{}".format(messages,chat_history_id,str(result_ds_collection_info.__dict__)))
    contents=[]
    for content in constants.text_generation_model.predict(chat_history_id,messages,max_token,temperature,result_ds_collection_info,retry_num=retry_num):
        contents.append(content)
        yield "".join(contents),chat_history_id
    logging.info("当前模型结束返回给gradio")
