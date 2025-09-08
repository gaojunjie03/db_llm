from sqlalchemy import create_engine
import jaydebeapi
from sqlalchemy.pool import NullPool
from sqlalchemy import text, inspect
import os
import pandas as pd
from urllib.parse import quote_plus as urlquote
from sqlalchemy.orm import sessionmaker
import json
from urllib.parse import  unquote

"""
    datasource结构如下：
    {
        "DATABASE_NAME":"", 数据库名
        "IP":"", 数据库ip
        "PORT":"", 数据库端口
        "USERNAME":"", 数据库账号
        "PASSWORD":"", 数据库密码
        "TYPE":"", 数据库类型，0 oracle， 1 mysql ，2 sqlite，3 达梦， 4 人大金仓，6 hive ， 11 pgsql 
    }
"""
DATABASE_DICT = {
    0: "oracle",
    1: "mysql",
    2: "sqlite",
    3: "dameng",
    4: "kingbase",
    6: "hive",
    11: "postgresql"
}


class Datasource():
    def __init__(self,DATABASE_NAME,IP,PORT,USERNAME,PASSWORD,TYPE,SCHEMA=None):
        if TYPE != 4 and TYPE != 3 :
            self.DATABASE_NAME = urlquote(DATABASE_NAME)
            self.USERNAME = urlquote(USERNAME)
            self.PASSWORD = urlquote(PASSWORD)
        else:
            self.DATABASE_NAME = DATABASE_NAME
            self.USERNAME = USERNAME
            self.PASSWORD = PASSWORD
        self.IP=IP
        self.PORT=PORT
        self.TYPE=TYPE
        if SCHEMA is None or SCHEMA == "":
            self.SCHEMA=self.DATABASE_NAME
        else:
            self.SCHEMA=SCHEMA
    def unquote(self):
        if self.TYPE != 4 and self.TYPE != 3 :
            self.DATABASE_NAME = unquote(self.DATABASE_NAME)
            self.USERNAME = unquote(self.USERNAME)
            self.PASSWORD = unquote(self.PASSWORD)
    def to_dict(self):
        return {
            "DATABASE_NAME": self.DATABASE_NAME,
            "IP": self.IP,
            "PORT": self.PORT,
            "USERNAME": self.USERNAME,
            "PASSWORD": self.PASSWORD,
            "TYPE": self.TYPE,
            "SCHEMA": self.SCHEMA
        }
        

database_url_mapper = {
    0: 'oracle://{username}:{password}@{ip}:{port}/?service_name={dbname}',
    1: 'mysql+pymysql://{username}:{password}@{ip}:{port}/{dbname}',
    11: 'postgresql://{username}:{password}@{ip}:{port}/{dbname}',
    2 : 'sqlite:///{dbname}'
}


def get_conn(datasource):

    def build_db_url(datasource):
        url_template = database_url_mapper.get(datasource.TYPE, None)
        if url_template is None:
            raise Exception(f'该数据库类型还没有适配, type:{datasource.TYPE}')
        return url_template.format(
            ip=datasource.IP,
            port=datasource.PORT,
            dbname=datasource.DATABASE_NAME,
            username=datasource.USERNAME,
            password=datasource.PASSWORD
        )

    if datasource.TYPE == 4 or datasource.TYPE == 3 or datasource.TYPE == 12 or datasource.TYPE == 6:
        dp=os.path.dirname(__file__)+"/drivers"
        driver_paths = [dp + "/kingbase8-8.6.0.jar",dp + "/DmJdbcDriver18.jar",dp + "/quark-driver-8.37.3.jar",dp + "/hive-jdbc-3.1.2.jar",
                        dp+"/libthrift-0.9.3.jar",dp+"/httpclient-4.4.1.jar",
                            dp + "/httpcore-4.4.1.jar",dp + "/slf4j-api-1.7.20.jar",dp + "/curator-client-2.6.0.jar",
                            dp + "/commons-lang-2.6.jar",dp + "/hive-exec-3.1.2.jar",
                            dp + "/hive-common-3.1.2.jar",dp + "/hive-service-3.1.2.jar",dp + "/hive-service-rpc-3.1.2.jar"]
        if datasource.TYPE == 4:
            driver = "com.kingbase8.Driver"
            url = "jdbc:kingbase8://{}:{}/{}".format(datasource.IP, datasource.PORT, datasource.DATABASE_NAME)
        elif datasource.TYPE == 3:
            driver = "dm.jdbc.driver.DmDriver"
            url = "jdbc:dm://{}:{}/{}".format(datasource.IP, datasource.PORT, datasource.DATABASE_NAME)
        elif datasource.TYPE == 6:
            driver = "org.apache.hive.jdbc.HiveDriver"
            url = "jdbc:hive2://{}:{}/{}?hive.resultset.use.unique.column.names=false".format(datasource.IP, datasource.PORT, datasource.DATABASE_NAME)
        return ConnectionWrapper(jaydebeapi.connect(driver, url, [datasource.USERNAME, datasource.PASSWORD], jars=driver_paths))
    else:
        engine = create_engine(build_db_url(datasource), poolclass=NullPool)
        return engine.connect()

class ConnectionWrapper(object):
    """
    Kingbase、DM的connection的包装类
    """
    def __init__(self, conn):
        self.conn = conn

    def execute(self,sql):
        cursor=self.conn.cursor()
        sql = sql.replace("`", "\"")
        cursor.execute(sql)
        return cursor

    def close(self):
        self.conn.cursor().close()
        self.conn.close()

    def cursor(self):
        return self.conn.cursor()

def get_tables_field(ds):
    conn=get_conn(ds)
    table_column_description_maps=[]
    if ds.TYPE in [4,3,12,6]:
        meta=conn.conn.jconn.getMetaData()
        rs = meta.getTables(None, ds.SCHEMA, "%", ['TABLE'])
        while rs.next():
            table_name = rs.getString("TABLE_NAME")
            table_comment = rs.getString("REMARKS")
            rs_cols = meta.getColumns(None, ds.SCHEMA, table_name, "%")
            columns = []
            while rs_cols.next():
                col_name = rs_cols.getString("COLUMN_NAME")
                col_type = rs_cols.getString("TYPE_NAME")
                remark=rs_cols.getString("REMARKS")
                columns.append({
                    "col_real_name":col_name,
                    "col_type":col_type,
                    "col_memo":remark
                })
            table_column_description_maps.append({
                "table_real_name":table_name,
                "description":table_comment,
                "columns":json.dumps(columns,ensure_ascii=False)
            })
    else:
        inspector=inspect(conn)
        tables=inspector.get_table_names()
        for table in tables:
            table_comment = inspector.get_table_comment(table)
            columns = []
            for col in inspector.get_columns(table):
                columns.append({
                    "col_real_name":str(col['name']),
                    "col_type":str(col['type']),
                    "col_memo":str(col['comment'])
                })
            table_column_description_maps.append({
                "table_real_name":str(table),
                "description":str(table_comment['text']),
                "columns":json.dumps(columns,ensure_ascii=False)
            })
    conn.close()
    return pd.DataFrame(table_column_description_maps)
        
    
        