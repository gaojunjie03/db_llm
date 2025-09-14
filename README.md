# 数据库智能查询系统

数据库智能查询系统。支持数据库元数据管理、表结构映射、向量化入库、以及基于大语言模型的自然语言智能查询，能够自动生成 SQL 并可视化查询结果。

---

## 功能特点
- **数据库元数据获取**：支持 MySQL、Oracle、PostgreSQL、达梦、Kingbase 数据库。  
- **智能表结构映射**：根据用户问题提取中文表描述，自动生成真实数据库表名与字段映射。  
- **向量化知识库构建**：将表描述向量化并存储至 Milvus，支持语义检索。  
- **自然语言智能查询**：用户用自然语言提问，系统自动生成 SQL 并返回查询结果。  
- **查询结果可视化**：支持 Markdown 表格渲染及统计图表生成。  
- **多阶段异常处理**：自动处理 SQL 执行异常、系统异常及函数调用异常，保证查询稳定性。  
- **安全机制**：自动拒绝可能破坏数据的操作（新增、修改、删除数据或表）。

---

## 技术特点
- **嵌入式向量模型**：采用 [Sentence-Transformers] 的 Qwen3-Embedding-0.6B 模型（本地部署）进行表结构向量化。  
- **语言生成模型**：使用 GPT-OSS-20B模型（本地部署）处理自然语言理解与 SQL 生成。  
- **消息流式解析**：采用 OpenAI Harmony 格式及 StreamableParser 技术，实现高效流式生成和增量输出。  
- **数据库与向量数据库支持**：MySQL、Oracle、PostgreSQL、达梦、Kingbase；向量数据存储在 Milvus，支持快速语义检索。  
- **前端交互**：基于 Gradio Blocks + ChatInterface。  
- **安全策略**：系统层面过滤高风险数据库操作，确保查询安全性。
---


## 安装与依赖

### 系统要求
- Python >= 3.12  
- CUDA 12.8+
- GPU 显存48G+ 
- CentOS / Ubuntu 等 Linux 系统  

### 安装依赖
```bash
pip install -r requirements.txt
cd models
git lfs clone https://huggingface.co/openai/gpt-oss-20b
git lfs clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

# 针对oracle
ubuntu：
apt-get install libaio1
centos:
yum install libaio
echo $(pwd)/db/drivers/oracle/instantclient* > /etc/ld.so.conf.d/oracle-instantclient.conf && ldconfig

# 采用jaydebeapi查询相关数据库如dm、kingbase，需要配置jdk环境
tar -zxvf jdk.tar.gz
export JAVA_HOME=$(pwd)/jdk1.8.0_192
export CLASSPATH=.:${JAVA_HOME}/lib/tools.jar:${JAVA_HOME}/lib/dt.jar
export PATH=${JAVA_HOME}/bin:/bin:/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
 ``` 
---

## 使用方法

### 1. 启动服务
在终端执行：
```bash
python gradio_app.py
 ```
### 2.浏览器访问 
```bash
http://127.0.0.1:6006
 ```
### 3.示意图
![image](https://github.com/user-attachments/assets/58dcd992-facc-424d-a7c8-29d4a78316ca)

![image](https://github.com/user-attachments/assets/549c9a87-4ff7-49ea-8647-6d4095af019e)

![image](https://github.com/user-attachments/assets/aef46f2c-e78a-43ab-8c83-69240aa68f12)

![image](https://github.com/user-attachments/assets/5e1af5f7-277e-483d-b941-c742dbbcf130)




