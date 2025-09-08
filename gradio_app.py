import extra_network
import constants
import os
from gradio_extra.api import show_columns,update_columns,insert_into_milvus,delete_tables,get_db_metadata,get_ds_collection_infos,chat_with_model
from gradio_extra.frontend import css,head
logging.info("加载嵌入式向量模型...")
sentence_model=SentenceModel()
constants.sentence_model=sentence_model
logging.info("加载语言生成模型...")
os.environ["TIKTOKEN_RS_CACHE_DIR"] = "./tiktoken"
text_generation_model=TextGenerationModel()
logging.info("初始化sqlite表信息...")
with open_sqlite(need_commit=True) as db:
   DsCollectionInfo.__table__.create(db.conn, checkfirst=True)



with gr.Blocks(css=css,head=head) as demo:
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