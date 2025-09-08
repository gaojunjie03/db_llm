from pymilvus import MilvusClient
from pymilvus import FieldSchema, CollectionSchema, DataType
from pymilvus import utility
from sqlalchemy.orm import sessionmaker
import db.dbutil as dbutil
from log.log import logging
import traceback
import os


class open_sqlite():
    def __init__(self,need_commit=False):
        self.need_commit=need_commit
    def __enter__(self):
        self.conn = dbutil.get_conn(dbutil.Datasource("data_sqlite.db","","","","",2,None))
        db_session = sessionmaker(bind=self.conn.engine)
        self.session = db_session()
        if self.need_commit:
            self.session.begin()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.need_commit:
            if exc_val is None:
                self.session.commit()
            else:
                self.session.rollback()
        if self.session is not None:
            self.session.close()
        if self.conn is not None  :
            self.conn.close()

        if exc_type is not None:
            logging.error(exc_type)
            raise Exception(str(exc_type))



class open_milvus():
    def __init__(self):
        pass
    def __enter__(self):
        self.client = MilvusClient(
            uri="data_milvus.db"
        )
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None  :
            self.client.close()
        if exc_type is not None:
            logging.error(exc_type)
            raise Exception(str(exc_type))



        
def create_table_knowledge_collection(collection_name,client):
    if not client.has_collection(collection_name=collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="table_real_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="table_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="columns", dtype=DataType.VARCHAR, max_length=1000),
        ]
        schema = CollectionSchema(fields=fields)
        client.create_collection(collection_name=collection_name, schema=schema)
        index_params = client.prepare_index_params()
        # FLAT 表示 暴力检索，也叫 Brute-Force Search。
        #检索时会对库里的所有向量逐一计算距离，所以精度最高，但速度最慢，内存消耗也大。
        #适合：小数据量（几万级以内），或者做基准测试。
        #metric_type="IP"
        #度量方式（向量相似度的计算方法）。
        #"IP" = Inner Product（内积/点积），常用于 余弦相似度 或 向量化推荐系统。适合语义搜索
        index_params.add_index(field_name="table_embedding", index_type="FLAT", metric_type="IP")
        client.create_index(collection_name, index_params)
        
#     def search_table_knowledge_collection(self,collection_name,embedding_data):
#         return self.client.search(
#             collection_name=collection_name,
#             data=embedding_data,
#             anns_field="table_embedding",
#             limit=3,
#             output_fields=["table_real_name","columns"],
#         )
        
