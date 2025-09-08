import gradio as gr
import pandas as pd
import db.dbutil as dbutil
import json
from sqlalchemy import text
from db.db_tools import open_sqlite,open_milvus,create_table_knowledge_collection
from models.db_models import SentenceModel,TextGenerationModel
from log.log import logging
from entity.ds_collection_info import DsCollectionInfo
import extra_network
import traceback
import uuid
import constants
import os
logging.info("加载嵌入式向量模型...")
sentence_model=SentenceModel()
constants.sentence_model=sentence_model
logging.info("加载语言生成模型...")
os.environ["TIKTOKEN_RS_CACHE_DIR"] = "./tiktoken"
text_generation_model=TextGenerationModel()
logging.info("初始化sqlite表信息...")
with open_sqlite(need_commit=True) as db:
   DsCollectionInfo.__table__.create(db.conn, checkfirst=True)



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
                embeddings = sentence_model.predict(batch["description"].tolist())
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
            text_generation_model.clear_chat_history(chat_history_id)
        chat_history_id=str(uuid.uuid1())
    logging.info("用户的消息:{},聊天记录id:{},数据库信息:{}".format(messages,chat_history_id,str(result_ds_collection_info.__dict__)))
    contents=[]
    for content in text_generation_model.predict(chat_history_id,messages,max_token,temperature,result_ds_collection_info,retry_num=retry_num):
        contents.append(content)
        yield "".join(contents),chat_history_id
    logging.info("当前模型结束返回给gradio")

with gr.Blocks(css="""
    /* 输入框、下拉框统一美化 */
    input, select, textarea {
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 15px !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    /* 聚焦时高亮边框 */
    input:focus, select:focus, textarea:focus {
        border: 1px solid #6a11cb !important;
        box-shadow: 0 0 6px rgba(106,17,203,0.3);
        outline: none !important;
    }
    
    /* Label 统一样式 
    label {
        font-weight: 600 !important;
        color: #333 !important;
        margin-bottom: 6px !important;
        display: block !important;
    }*/
    
    /* 表格样式 */
    .gradio-dataframe table {
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 14px;
    }
    
    /* 表格表头 */
    .gradio-dataframe thead {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        font-weight: bold;
    }
    
    /* 表格单元格 */
    .gradio-dataframe td, 
    .gradio-dataframe th {
        padding: 10px 14px;
        border: 1px solid #eee;
    }
    
    /* 表格 hover 效果 */
    .gradio-dataframe tbody tr:hover {
        background-color: #f9f9ff;
    }


    .fancy-btn {  background: linear-gradient(135deg, #6a11cb, #2575fc); color: white !important;
    font-weight: bold;
    font-size: 16px;
    padding: 12px 24px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);}
    
    .fancy-btn:hover {
    background: linear-gradient(135deg, #2575fc, #6a11cb);
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
    
    .fancy-btn:active {
        transform: scale(0.95);
    }
    
    .green-btn {
       background: linear-gradient(135deg, #2e7d32, #66bb6a);
    }
    .selection-button {
        display:none;
    }
    .cell-menu-button {
       display:none;
    }
    .cell-selected{
      --ring-color: none;
    }
    .add-row-button {
       display:none;
    }
    /* 先隐藏所有按钮 */
    .cell-menu button {
      display: none;
    }
    
    /* 只保留第 3 个 */
    .cell-menu button:nth-child(3) {
      display: inline-block;  /* 或者 flex，看原来的布局 */
    }
    
    .red-btn {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
    }
    """,head="""
    <script>
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.tagName === "TEXTAREA") {
                      // 找到它所在的 td
                    const td = node.closest("td");
                    // 找到所在的 tr
                    const tr = node.closest("tr");
                    if (!td || !tr) return;
    
                    // 获取所有列
                    const tds = Array.from(tr.children);
                    // 如果不是最后一列，则只读
                    if (td !== tds[tds.length - 1]) {
                        node.readOnly = true;
                    } else {
                        node.readOnly = false; // 最后一列可编辑
                    }
                }
                
            });
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """) as demo:
    gr.Markdown("""
    # ....
    **Tips**
    1. ....
    """)
    with gr.Accordion("数据知识库设置", open=False):  
        with gr.Row():
            with gr.Column(scale=2):
                ds_type = gr.Dropdown(choices=[
                ("Oracle 数据库", 0),   
                ("MySQL 数据库", 1),
                ("达梦数据库",3),
                ("Kingbase 数据库", 4),
                ("PostgreSQL 数据库", 11),
                ("Hive 数据库", 6)],value=1, label="请选择数据库类型",interactive=True)
                database_ip = gr.Textbox(label="请输入数据库ip",value="8.148.245.50")
                database_port = gr.Number(label="请输入数据库端口",value="3306")
                database_name = gr.Textbox(label="请输入数据库名",value="jxmpc")
                database_schema = gr.Textbox(label="请输入数据库模式",value="jxmpc")
                database_username = gr.Textbox(label="请输入数据库登录账号",value="root")
                database_password = gr.Textbox(label="请输入数据库登录密码",value="JxmpcServer!@#123")
                metadata_btn=gr.Button("获取元数据", elem_classes="fancy-btn green-btn")
                milvus_collection_name = gr.Textbox(label="入库表名", visible=False)
                milvus_collection_description = gr.Textbox(label="描述用途", visible=False)
                insert_milvus_btn=gr.Button("元数据信息入库", elem_classes="fancy-btn red-btn", visible=False)
                info_box_markdown = gr.Markdown("", visible=False)
            with gr.Column(scale=2):
                tables_df=gr.Dataframe(label="数据库表信息",interactive=True,headers=["数据库表名", "描述"],col_count=2)
            with gr.Column(scale=3):
                columns_df = gr.Dataframe(
                    label="对应表元数据信息",
                    headers=["字段名","类型","备注"],
                    interactive=True,col_count=3
                )
                field_state = gr.State([])
                table_state = gr.State([])
                current_table_index_state=gr.State()
    
                
            tables_df.change(delete_tables,inputs=[current_table_index_state,tables_df,table_state,field_state,columns_df],outputs=[table_state,field_state,columns_df])    
            tables_df.select(show_columns, inputs=[field_state,table_state], outputs=[columns_df,current_table_index_state])
            columns_df.input(update_columns,inputs=[field_state,columns_df,current_table_index_state], outputs=[columns_df,field_state])
            metadata_btn.click(get_db_metadata,inputs=[ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password],outputs=[tables_df,field_state,table_state,milvus_collection_name,milvus_collection_description,insert_milvus_btn])
            insert_milvus_btn.click(insert_into_milvus,inputs=[milvus_collection_name,milvus_collection_description,ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password,tables_df,field_state],outputs=[info_box_markdown])
    with gr.Accordion("数据库智能查询", open=False):  
         with gr.Row():
             with gr.Column(scale=1):
                 ds_collection_infos_select = gr.Dropdown(choices=[],label="请选择查询的数据库信息",interactive=True)
                 get_ds_collection_infos_btn=gr.Button("获取数据库信息", elem_classes="fancy-btn red-btn", visible=True)
                 max_token_slider = gr.Slider(label="调整token数", minimum=500, maximum=3000, value=1000, step=100 )
                 temperature_slider = gr.Slider(label="调整随机性，越低越准确，越高越有创造性",
                                                minimum=0.05, maximum=1.0, value=0.05, step=0.05 )
                 retry_slider = gr.Slider(label="设置异常重试次数(系统异常、sql执行异常等)",
                                                minimum=3, maximum=10, value=3, step=1) 
                 ds_collection_info_state=gr.State([])
             with gr.Column(scale=4):
                 chat_history_id=gr.State()
                 gr.ChatInterface(
                        chat_with_model,
                        type="messages",
                        chatbot=gr.Chatbot(height=800),
                        additional_inputs=[ds_collection_infos_select,ds_collection_info_state,max_token_slider,temperature_slider,chat_history_id,retry_slider],
                        additional_outputs=[chat_history_id],
                        examples=[["查询一下2025年4月份之间有哪些项目是develop这个用户创建的，返回项目名称、格式化后计算模式类型、创建时间(这个请格式化成YYYY/MM/DD HH:mm:ss格式)"]],
                        textbox=gr.Textbox(placeholder="请用自然语言描述查询需求，例如：电子类订单总额超过500w并且属于广东地区的男性用户有哪些", container=False, scale=7)
                 )
                 

         get_ds_collection_infos_btn.click(get_ds_collection_infos,inputs=[],outputs=[ds_collection_infos_select,ds_collection_info_state])
        
             
             
        
    
        
demo.launch(share=True,server_port=6006)