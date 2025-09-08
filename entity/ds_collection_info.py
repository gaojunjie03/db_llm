from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class DsCollectionInfo(Base):
    __tablename__ = "ds_collection_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_name = Column(String)
    description = Column(String)
    datasource = Column(String)


    