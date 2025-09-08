from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,StoppingCriteria
from abc import abstractmethod
from log.log import logging
from transformers import TextIteratorStreamer
from threading import Thread
import datetime
from queue import Queue
import json
import traceback
import functions_call as functions
from  entity.ds_collection_info import DsCollectionInfo
from typing import List
import re
from threading import Lock
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    StreamableParser,
    Author,
    DeveloperContent,
    HarmonyError,
    ReasoningEffort,
    RenderConversationConfig,
    SystemContent,
    ToolDescription
)
from datetime import datetime



class DbModel:
    def __init__(self,model,encoding):
        self.model=model
        self.encoding=encoding
        
    @abstractmethod
    def predict(self,*args,**kwargs):
        pass
        
        
class SentenceModel(DbModel):
    def __init__(self):
        embedding_model_name = "./models/Qwen3-Embedding-0.6B"
        embedding_model = SentenceTransformer(embedding_model_name)
        super(SentenceModel,self).__init__(embedding_model,None)

    def predict(self,datas:List[str]=None):
        return self.model.encode(datas)


class MyStop(StoppingCriteria):
    def __init__(self, stop_flag):
        self.stop_flag = stop_flag

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_flag["stop"]

class HarmonyStreamer:
    """
    可替代 TextIteratorStreamer，增量解析 Harmony 消息。
    """
    def __init__(self, encoding,stage_index=None):
        self.generated_ids = []
        self.finished = False
        self.lock = Lock()
        self.callbacks = []
        self.stage_index=stage_index
        self.stream = StreamableParser(encoding, role=Role.ASSISTANT)

    def put(self, token_id):
        if token_id.numel() == 1:
            token_id = token_id.tolist() 
        else:
            token_id = token_id[0].tolist()
        with self.lock:
            for token in token_id:
                try:
                    self.stream.process(token)
                except Exception as e:
                    logging.error(str(e))
                    continue
              
                for cb in self.callbacks:
                    cb({
                        "current_role": self.stream.current_role,
                        "current_channel": self.stream.current_channel,
                        "last_content_delta":self.stream.last_content_delta,
                        "current_content_type":self.stream.current_content_type,
                        "current_recipient": self.stream.current_recipient,
                        "current_content": self.stream.current_content
                    })

    def end(self):
        with self.lock:
            logging.info("阶段{}模型结束输出".format(self.stage_index))
            self.finished = True

    def subscribe(self, callback):
        self.callbacks.append(callback)


class TextGenerationModel(DbModel):
    def __init__(self):
        model_name = "./models/gpt-oss-20b"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        super(TextGenerationModel,self).__init__(model,encoding)
        self.messages_history={}
        self.dev = (
                DeveloperContent.new()
                .with_instructions("调用query_table前需要先调用map_tables_fieds获取表结构，调用plot_charts函数则需要调用query_table之后得到数据并且经过用户同意再调用plot_charts函数")
                .with_function_tools(
                    [
                        ToolDescription.new(
                            "map_tables_fields",
                            "根据分析用户问题得到的各表中文描述获取各表对应的真实数据库表名和数据库表字段信息",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "table_name":{
                                                   "type":"string",
                                                   "description": "中文描述的数据表名",
                                                }
                                            },
                                             "required": ["table_name"],
                                        },
                                        "description": " 需要映射的表名列表，使用中文描述",
                                    }
                                },
                                "required": ["parameters"],
                            },
                        ),
                        ToolDescription.new(
                            "query_table",
                            "执行sql语句",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "sql":{
                                                   "type":"string",
                                                   "description": "sql语句",
                                                }
                                            },
                                            "required": ["sql"]
                                        },
                                        "description": "参数对象",
                                    },
                                    "required": ["parameters"],
                                },
                        ),
                        ToolDescription.new(
                            "plot_charts",
                            "生成统计图表",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "title":{
                                                   "type":"string",
                                                   "description": "统计图表的标题",
                                            },
                                            "bar_xlabel":{
                                                   "type":"string",
                                                   "description": "横坐标名称",
                                            },
                                            "bar_ylabel":{
                                                   "type":"string",
                                                   "description": "纵坐标名称",
                                            },
                                            "label":{
                                                  "type":"array",
                                                  "description": "标签数据",
                                                   "items": {
                                                        "type": "string",
                                                    },
                                            },
                                            "data":{
                                                  "type":"array",
                                                  "description": "各标签对应的统计数据",
                                                   "items": {
                                                        "type": "number",
                                                    },
                                            },
                                            "required": ["title","label","data","bar_xlabel","bar_ylabel"]
                                        },
                                        "description": "参数对象",
                                    },
                                    "required": ["parameters"],
                                },
                            },
                        ),
                    ]
                )
            )
    def match_message(self,reply,rule):
        match = re.search(rule, reply)
        content = match.group(1) if match else reply
        return match,content
    def make_sys(self,current_date):
        self.sys=(
                SystemContent.new()
                .with_model_identity(
                    self.system_prompt()
                )
                .with_reasoning_effort(ReasoningEffort.LOW)
                .with_conversation_start_date(current_date)
                .with_knowledge_cutoff("2025-09-01")
        )
    def predict(self,chat_history_id:str, user_input:str,max_new_tokens=2048,temperature=0.05,ds_collection_info:DsCollectionInfo =None,tool_name :str=None,retry_num:int=4,stage_run_status:bool=True):
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.make_sys(current_date)
        if chat_history_id not in self.messages_history:
            self.messages_history[chat_history_id]=[]
        stage_index = 1 
        last_stage = False
        error_obj_count={}
        while True:
            if tool_name is not None:
                self.messages_history[chat_history_id].append({"role":"tool","content":user_input,"tool_name":tool_name})
            else:
                self.messages_history[chat_history_id].append({"role":"user","content":user_input})

            
            stream = StreamableParser(self.encoding, role=Role.ASSISTANT)
            

            messages_harmony=[Message.from_role_and_content(Role.SYSTEM,self.sys),
                Message.from_role_and_content(Role.DEVELOPER, self.dev)]
            for m in self.messages_history[chat_history_id]:
                mrole=m["role"]
                if mrole=="user":
                    role=Role.USER
                elif mrole=="assistant":
                    role=Role.ASSISTANT
                elif mrole=="tool":
                    role=Role.TOOL

                if mrole=="user" or mrole=="assistant":
                    messages_harmony.append(Message.from_role_and_content(role, m["content"]))
                else:
                    messages_harmony.append(Message.from_author_and_content(
                        Author.new(role, m["tool_name"]),
                        m["content"],
                    )
                    .with_channel("commentary")
                    .with_recipient("assistant"))

            convo = Conversation.from_messages(messages_harmony)

            input_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            streamer = HarmonyStreamer(self.encoding,stage_index=stage_index)
            q = Queue()
            streamer.subscribe(lambda msg: q.put(msg))
            stop_flag = {"stop": False}
            generate_dict=dict(
                inputs= torch.tensor([input_ids], dtype=torch.long).to(self.model.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature, 
                streamer=streamer,
                stopping_criteria=[MyStop(stop_flag)]
            )
        
            thread = Thread(target=self.model.generate, kwargs=generate_dict)
            thread.start()
            last_answer = ""
            idx = 0
            thinking = True
            commentary = False
            last_channel = ""
            commentary_name=None
            commentary_content=None
            def get_commentary_tool(name,content):
                logging.info("阶段{}最终的函数调用{}{}".format(stage_index,name,content))
                validate_flag,content_body=validate_json(content)
                if not validate_flag:
                    return validate_flag,content_body
                content_body["method"]=name
                return validate_flag,json.dumps(content_body,ensure_ascii=False)
            def validate_json(content):
                try:
                    content_body=validate_json_deep(content)
                    return True,content_body
                except Exception as e:
                    return False,content+",解析该json结构失败！"+str(e)
            def validate_json_deep(content):
                try:
                    content_body=json.loads(content.strip())
                except Exception:
                    content_body=json.loads(content.strip()[:-1])
                return content_body
            yield f"\n<details open>\n"
            yield f"\n<summary style=\"font-size:18px; color:darkred;\">开始阶段{stage_index}分析</summary>\n"
            yield f"\n<div style=\"font-size:14px; line-height:1.6;\">\n"           
            while not streamer.finished:
                try:
                    msg = q.get(timeout=0.1)
                except:
                    continue

                current_role = msg["current_role"]
                last_content_delta = msg["last_content_delta"]
                current_recipient = msg["current_recipient"]
                current_channel = msg["current_channel"]
                current_content=msg["current_content"]
                if current_recipient is not None and current_recipient.endswith("system"):
                    continue
                if current_channel is not None:
                    if last_channel == "commentary" and current_channel != "commentary":
                        val_flag,_=validate_json(commentary_content)
                        if not val_flag:
                            logging.info("阶段{}未完成生成完整的函数参数json，继续生成,当前channel{}".format(stage_index,current_channel))
                        else:
                            logging.info("阶段{}已经生成过一次函数调用无需再生成，当前channel{}，停止模型输出".format(stage_index,current_channel))
                            stop_flag["stop"] = True
                    if current_channel == "commentary" and current_recipient != "assistant" :
                        # logging.info("当前频道是{},需要调用的函数是{}".format(current_channel,current_recipient))
                        last_channel=current_channel
                        if not commentary:
                            commentary_name=current_recipient
                            yield current_recipient
                        commentary=True
                        if last_content_delta is not None:
                            yield last_content_delta
                        commentary_content=current_content
                    elif current_channel == "final" and last_content_delta is not None :
                        # logging.info("当前频道是{},返回的结果是{}".format(current_channel,current_content))
                        if thinking:
                            yield f"\n</div>\n"
                            yield f"\n</details>\n"
                        thinking=False
                        last_answer +=last_content_delta
                        if commentary_name is None :
                            yield last_content_delta
                    elif current_channel == "analysis" and last_content_delta is not None :
                        yield last_content_delta
                elif last_channel == "commentary":
                    # 这里需要确认一下是否已经生成了完整的参数json
                    val_flag,_=validate_json(commentary_content)
                    if not val_flag:
                        logging.info("阶段{}未完成生成完整的函数参数json，继续生成,当前channel None".format(stage_index))
                    else:
                        logging.info("阶段{}已经生成过一次函数调用无需再生成，当前channel None，停止模型输出".format(stage_index))
                        stop_flag["stop"] = True
                idx = idx + 1
                
            # 必须要线程停止运行之后再进行下方处理
            logging.info("阶段{}流式输出线程等待其停止运行".format(stage_index))
            thread.join()
            logging.info("阶段{}流式输出线程已停止".format(stage_index))
            if not stage_run_status:
                # 上一个阶段执行失败（可能函数执行失败或者系统异常导致的重试），
                # 其中函数执行失败会将失败信息设置成user_input给到tool这个角色进行让模型分析，
                # 为了避免后续模型重复对历史记录的失败信息再次分析，当当前阶段执行成功需要将上一个异常信息从记录中剔除
                # 而系统异常则会user_input原封不动的重新让模型分析，这会造成重复的聊天记录，因此这里也将记录中剔除当前user_input
                # 这里不管当前阶段执行成功还是失败都要剔除当前失败的聊天记录，让模型每次回答都专注于当前。
                self.messages_history[chat_history_id] = self.messages_history[chat_history_id][:-1] 
                logging.info("阶段{},上一个阶段出现失败,剔除当前模型输入".format(stage_index))
            else:
                # 上一个阶段成功，则保留成功的聊天记录信息
                pass
            if last_stage:
                # 达到了最后一个阶段，即sql执行查询数据库成功并返回数据table
                break
            yield "\n"
            try:
                validate_flag = None
                commentary_content_json = None
                if commentary_name is not None and commentary_content is not None:
                    validate_flag,commentary_content_json=get_commentary_tool(commentary_name,commentary_content)
                if validate_flag is not None and not validate_flag and commentary_name is not None and commentary_content is not None:
                    logging.info("阶段{}解析functions的参数失败{}".format(stage_index,commentary_content_json))
                    # user_input=json.dumps({
                        # "msg":"解析json失败,请重新生成当前function调用的参数"
                    # }, ensure_ascii=False)
                    # tool_name=commentary_name
                    # stage_run_status=False
                    raise functions.FunctionsException("解析json失败,请重新生成当前function调用的参数")
                else:
                    # 5. 保存最终答案到历史
                    if commentary_content_json is not None:
                        last_answer=commentary_content_json
    
                    logging.info("阶段{}解析last_answer:{}".format(stage_index,last_answer))
                    try:
                        method_param = json.loads(last_answer.strip())
                    except BaseException as e:
                        # 这里说明用户在跟模型闲聊，直接break
                        break
                    if "code" in method_param and method_param["code"] == "error":
                        logging.info("last_answer有异常信息:{}".format(method_param["msg"]))
                        yield "\n"+method_param["msg"]
                        break
                    method = method_param["method"]
                    parameters = method_param["parameters"]
                    logging.info("阶段{}执行函数{}".format(stage_index,method))
                    yield f"\n阶段{stage_index}正在执行函数中..."
                    tmp_user_input,tmp_last_stage = eval(method+"(parameters,ds_collection_info)")
                    yield f"\n阶段{stage_index}执行完成..."
                    logging.info("阶段{}函数{}执行完成".format(stage_index,method))
                    yield f"\n阶段{stage_index}函数执行结果:\n"
                    if "plot_charts" in method:
                        # 由于生成的图表base64太长，导致非常容易爆显存，因此这里就不交给模型了，直接渲染给gradio
                        yield tmp_user_input
                        break
                    for u in list(tmp_user_input):
                        yield u
                    user_input=tmp_user_input
                    last_stage=tmp_last_stage
                    tool_name = method
                    stage_run_status=True
            except functions.FunctionsException as e:
                logging.error("阶段{}执行functions{}异常:{}".format(stage_index,commentary_name,traceback.format_exc()))
                # 将当前function执行失败的信息当作user_input然后采用tool这个角色再次调用模型告诉模型functions调用失败，失败原因
                stage_run_status=False
                tool_name = commentary_name
                last_stage=False
                user_input=json.dumps({
                    "msg":str(e)
                }, ensure_ascii=False)
                if commentary_name not in error_obj_count:
                    error_obj_count[commentary_name]=1
                if error_obj_count[commentary_name] > retry_num:
                    yield f"\n执行functions{commentary_name}异常次数已达{error_obj_count[commentary_name]}次，停止输出\n"
                    break
                error_obj_count[commentary_name] += 1 
                yield f"\n阶段{stage_index}执行functions{commentary_name}异常:{str(e)}，尝试解决异常\n"
            except BaseException as e:
                logging.error(traceback.format_exc())
                if stage_index > retry_num:
                    yield f"\n系统异常次数已达{stage_index}次，停止输出\n"
                    break
                stage_run_status=False
                yield f"\n阶段{stage_index}系统出现异常，开始重试\n"
            finally:
                stage_index = stage_index+1
                yield f"\n</div>\n"
                yield f"\n</details>\n"
    def clear_chat_history(self,chat_history_id:str):
        self.messages_history.pop(chat_history_id, None) 
    def system_prompt(self):
        # system_prompt=
        # """
        #     你是数据库查询智能体助手
        #     必须严格遵循以下协议：
        #     1、分析用户问题，提取涉及的数据表，数据表名必须使用中文描述，这个中文描述你可以从用户问题中提取，可以进行猜测表中文名。最终调用functions.map_tables_fields获取映射表，由系统执行，你可以直接停止输出任何消息到final通道。
        #     2、得到映射表，如果映射表已经满足用户的问题后可以进行生成sql，否则你需要再次进行第1点获取其他还没得到映射的数据表结构，切记你生成的sql一定要通过分析映射表得到真实表名和真实字段名后再去生成。sql语句请采用分页查询，每页5条，查询第一页，sql语句不能含有任何的换行符，select每个字段都as起别名，别名使用用户问题中要返回的中文名。最终调用functions.query_table，由系统执行sql语句，你可以直接停止输出任何消息到final通道。
        #     3、关于动态处理新增/修改条件，若用户发出新的查询请求，在你已经拥有的映射表中如果不存在关于新的查询请求中的数据表和字段，则请分析新的查询请求中新的数据表中文描述重新执行第1点，如果你有不清楚的数据表中文描述你就直接问用户，让用户给你提示还有什么可用的数据表中文描述，然后你再根据用户的数据表中文描述去执行第1点工作获取映射表，而如果已经得到映射的表和字段则无需重复提取。
        #     4、functions.query_table函数可能会返回相关sql执行失败的异常信息给你，请麻烦分析异常信息，如果报错信息涉及语法错误，请对比异常信息检查你生成的sql是否有问题，重新执行第2点。
        #     5、functions.query_table函数执行成功会返回一个数组的json结构给你，如果是一个空的数组json如"[]",说明没有数据，则直接在final通道输出"无数据"，否则就说："以下是查询结果："，然后请直接将这个数组json结构渲染成markdown的表格形式，然后同时对表格内的数据做简单总结，然后在最后询问一下用户是否需要生成统计图表，请你注意你不要询问用户要生成什么图表，你就直接说是否需要生成图表，最终在final通道输出。
        #     6、在第5点，如果经过用户同意生成统计图表，得到用户的肯定答复，则你分析一下第5点的json数据，构成可生成图表的数据结构调用functions.plot_chart函数。
        #     7、用户的问题如果涉及对数据库表的新增数据、修改数据、删除数据、新增数据表、删除数据表、修改数据表这6种数据库行为，请直接在final通道输出：{"code":"error","msg":"对不起，此数据库操作存在数据安全风险，我不予执行"}，但是请你注意其他行为不要返回此信息。
        # """
        system_prompt="""
        你是数据库查询智能体助手  
        必须严格遵循以下协议：  
        1、分析用户问题，提取涉及的数据表，数据表名必须使用中文描述，这个中文描述你可以从用户问题中提取，也可以合理猜测。最终调用 functions.map_tables_fields 获取映射表，由系统执行。
        2、得到映射表后，如果映射表已经满足用户的问题，则生成 SQL；否则请回到第1点获取缺失的数据表结构。  
           - SQL 必须通过分析映射表得到真实表名和真实字段名。  
           - SQL 语句必须为分页查询，每页 5 条，查询第一页。  
           - SQL 语句不能包含换行符。  
           - SELECT 中每个字段都要加 `AS` 别名，别名使用用户问题中要求返回的中文名。  
           - 最终调用 functions.query_table，由系统执行 SQL 语句。
        3、当用户发出新的查询请求时：  
           - 如果新请求中涉及的数据表或字段不在现有映射表中，请分析请求中新的数据表中文描述，重新执行第1点。  
           - 如果所需数据表和字段已经存在映射，则直接基于现有映射继续执行第2点。  
        4、functions.query_table 可能返回 SQL 执行失败的异常信息：  
           - 如果报错涉及语法错误，请对照报错信息检查并修正 SQL，然后重新执行第2点。  
        5、functions.query_table 执行成功后会返回一个 JSON 数组：  
           - 如果结果为 `[]`（空数组），则在 final 通道直接输出：`无数据`，并结束流程。  
           - 如果结果非空，则在 final 通道输出：  
             - “以下是查询结果：”  
             - 把 JSON 渲染为 Markdown 表格  
             - 给出 1-2 句简要总结  
             - 最后询问用户：
               **“是否需要生成统计图表？”**
             - 对用户回复的判定规则：
                 - **肯定意图**（任一匹配即视为肯定）：`^(?:是|要|需要|可以|好|行|ok|yes|y|生成|画图|展示|来个图|需要生成|需要图表|可视化|统计)$` （忽略大小写及前后空白）
                 - **否定意图**（任一匹配即视为否定）：`^(?:否|不需要|不用|不要|不|no|n|算了|下次|以后再说)$` （忽略大小写及前后空白）
                 - **对于未匹配到上述任何意图的回复**，在 final 通道输出：`"抱歉，我没有理解您的意思。如果您需要生成图表，请回复‘是’或‘需要’；如果不需要，请回复‘否’或‘不用’。"` 并等待用户下一次输入，根据新输入再次进行意图判定。
        6、当且仅当用户回复被判定为肯定时：  
           - 分析第5点得到的 JSON 数据  
           - 构造可用于绘制图表的数据结构  
           - 调用 functions.plot_charts，并将数据传入  
           如果用户回复为否或未确认，则 final 通道输出：“好的，不生成图表。” 并结束流程。  
        7、如果用户的问题涉及以下 6 种数据库行为：新增数据、修改数据、删除数据、新增数据表、删除数据表、修改数据表，则在 final 通道直接输出：  
           `{"code":"error","msg":"对不起，此数据库操作存在数据安全风险，我不予执行"}`  
           - **判断标准**：用户问题必须包含直接指示这些操作的关键词（如“插入”、“更新”、“删除”、“创建表”、“删除表”、“修改表”），且上下文明确是数据操作而非查询。如果不确定，默认不输出错误信息。  
        """
        return system_prompt
