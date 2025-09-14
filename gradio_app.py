import extra_network
import constants
import gradio as gr
import os
from gradio_extra.api import show_columns,update_columns,insert_into_milvus,delete_tables,get_db_metadata,get_ds_collection_infos,chat_with_model,change_database_info,delete_ds_collection_info
from gradio_extra.frontend import css,head
from log.log import logging
from models.db_models import SentenceModel,TextGenerationModel
from db.db_tools import open_sqlite
from entity.ds_collection_info import DsCollectionInfo
logging.info("加载嵌入式向量模型...")
sentence_model=SentenceModel()
constants.sentence_model=sentence_model
logging.info("加载语言生成模型...")
os.environ["TIKTOKEN_RS_CACHE_DIR"] = "./tiktoken"
text_generation_model=TextGenerationModel()
constants.text_generation_model=text_generation_model
logging.info("初始化sqlite表信息...")
with open_sqlite(need_commit=True) as db:
   DsCollectionInfo.__table__.create(db.conn, checkfirst=True)



with gr.Blocks(css=css,head=head) as demo:
    gr.Markdown("""
    ## 数据库智能查询系统
     - **数据知识库设置** 
     - **数据库智能查询**
    """)
    with gr.Accordion("数据知识库设置", open=False): 
        with gr.Accordion("使用说明", open=False): 
            gr.Markdown("""
                - **选择数据库**  
                  支持多种数据库类型：Oracle、MySQL、PostgreSQL、达梦、Kingbase。        
                - **信息录入**  
                  输入目标数据库的关键信息，包括：IP 地址、端口号、数据库模式、数据库名、用户名和密码。         
                - **获取元数据**  
                  点击“获取元数据结构”按钮，将自动加载并展示指定数据库中的所有表结构。你可以在右侧表格中查看这些结构。   
                - **编制表描述信息**  
                  为了确保模型能准确进行数据表相似性检索，请尽可能准确地编辑每个数据表的描述信息。清晰的描述有助于提高模型查找相关表的精度。    
                - **编制表字段备注**  
                  数据字段的备注信息对模型理解数据结构至关重要。虽然字段信息不会直接录入向量数据库，但建议准确填写备注，尤其是字段为字典类型时。例如：“运行状态：1=运行中, 2=停止, 3=运行成功, 4=运行失败”。 
                - **删除无用表或字段**  
                  如遇到不需要查询的表或字段，可以通过右键删除来精简知识库内容。这样不仅减少了存储空间，还有效降低了GPU显存占用，使模型更加聚焦于关键数据。
                - **元数据信息入库**  
                  完成上述步骤后，可以将表的元数据信息入库到向量数据库。在填写入库表名时，确保使用英文+数据作为表名，方便向量数据库的管理与查询。输入表的描述信息以提供友好的展示，并点击“元数据信息入库”按钮完成操作。
                - **修改已入库的数据库信息**
                  可以随时对已入库到向量数据库的数据库信息进行修改。
                """)
        with gr.Row():
            with gr.Column(scale=2):
                already_all_ds_collection_info_state=gr.State([])
                already_insert_db_info = gr.Dropdown(choices=[],label="选择已入库的数据库信息进行修改",interactive=True)
                refresh_already_ds_collection_infos_btn=gr.Button("刷新已入库数据库信息", elem_classes="fancy-btn green-btn", visible=True)
                delete_ds_collection_infos_btn=gr.Button("删除当前数据库信息", elem_classes="fancy-btn red-btn", visible=False)
                delete_box_markdown = gr.Markdown("", visible=False)
                already_insert_db_info_select_state=gr.State()
                ds_type = gr.Dropdown(choices=[
                ("Oracle 数据库", 0),   
                ("MySQL 数据库", 1),
                ("达梦数据库",3),
                ("Kingbase 数据库", 4),
                ("PostgreSQL 数据库", 11)],value=1, label="请选择数据库类型",interactive=True)
                database_ip = gr.Textbox(label="请输入数据库ip")
                database_port = gr.Number(label="请输入数据库端口")
                database_name = gr.Textbox(label="请输入数据库名")
                database_schema = gr.Textbox(label="请输入数据库模式")
                database_username = gr.Textbox(label="请输入数据库登录账号")
                database_password = gr.Textbox(label="请输入数据库登录密码")
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
            already_insert_db_info.change(change_database_info,inputs=[already_insert_db_info,already_all_ds_collection_info_state],outputs=[ds_type,database_ip,database_port,database_name,
database_schema,database_username,
database_password,milvus_collection_name,milvus_collection_description,already_insert_db_info_select_state,tables_df,field_state,table_state,insert_milvus_btn,metadata_btn,delete_ds_collection_infos_btn])   
            tables_df.change(delete_tables,inputs=[current_table_index_state,tables_df,table_state,field_state,columns_df],outputs=[table_state,field_state,columns_df])    
            tables_df.select(show_columns, inputs=[field_state,table_state], outputs=[columns_df,current_table_index_state])
            columns_df.input(update_columns,inputs=[field_state,columns_df,current_table_index_state], outputs=[columns_df,field_state])
            metadata_btn.click(get_db_metadata,inputs=[ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password],outputs=[tables_df,field_state,table_state,milvus_collection_name,milvus_collection_description,insert_milvus_btn])
            insert_milvus_btn.click(insert_into_milvus,inputs=[milvus_collection_name,milvus_collection_description,ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password,tables_df,field_state],outputs=[info_box_markdown,milvus_collection_name,milvus_collection_description,ds_type,database_ip,database_port,database_name,database_schema,database_username,database_password,tables_df,field_state,columns_df,insert_milvus_btn])
            refresh_already_ds_collection_infos_btn.click(get_ds_collection_infos,inputs=[],outputs=[already_insert_db_info,already_all_ds_collection_info_state])
            delete_ds_collection_infos_btn.click(delete_ds_collection_info,inputs=[already_insert_db_info_select_state],outputs=[delete_box_markdown,already_insert_db_info,already_all_ds_collection_info_state])
    with gr.Accordion("数据库智能查询", open=False):  
        with gr.Accordion("使用说明", open=False): 
             gr.Markdown("""
               - **选择查询数据库**  
                  点击“获取数据库信息”按钮，获取需要查询的数据库列表，根据实际业务需求选择要查询的目标数据库。         
               - **参数设置**  
                  调整以下参数以优化查询：
                  - 最大 token 数：设置查询过程中允许的最大 token 数量。
                  - 随机性：控制查询结果的多样性和变化。
                  - 异常重试次数：当发生系统或数据库执行异常时，设置允许的重试次数。            
               - **对话智能查询**  
                  使用自然语言描述查询请求。例如：  
                  “查询在2025年08月注册并在订单中购买过服装类商品的男性用户，且这些用户的购买总额超过5000元。”
                  """)
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
                 chat_interface=gr.ChatInterface(
                        chat_with_model,
                        type="messages",
                        chatbot=gr.Chatbot(height=1000),
                        additional_inputs=[ds_collection_infos_select,ds_collection_info_state,max_token_slider,temperature_slider,chat_history_id,retry_slider],
                        additional_outputs=[chat_history_id],
                        textbox=gr.Textbox(placeholder="请用自然语言描述查询需求", container=False, scale=7)
                 )
            get_ds_collection_infos_btn.click(get_ds_collection_infos,inputs=[],outputs=[ds_collection_infos_select,ds_collection_info_state])
demo.launch(share=True,server_port=6006)