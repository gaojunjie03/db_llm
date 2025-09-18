# 基于gpt-oss模型harmony格式的数据库智能查询系统

数据库智能查询系统。支持数据库元数据管理、表结构映射、向量化入库、以及基于大语言模型的自然语言智能查询，能够自动生成 SQL 并可视化查询结果。

(This is my first open-source project after stepping into AI large language models. It’s still pretty simple, but every star means a lot to me. Thanks for your support!)

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
- GPU 显存48G+（16G显存可启动系统，但后续的对话，显存会不断增加）
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
## 1、数据知识库设置
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
![image](https://github.com/user-attachments/assets/58dcd992-facc-424d-a7c8-29d4a78316ca)

## 2. 数据库智能查询
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

![image](https://github.com/user-attachments/assets/549c9a87-4ff7-49ea-8647-6d4095af019e)

![image](https://github.com/user-attachments/assets/aef46f2c-e78a-43ab-8c83-69240aa68f12)

![image](https://github.com/user-attachments/assets/5e1af5f7-277e-483d-b941-c742dbbcf130)




