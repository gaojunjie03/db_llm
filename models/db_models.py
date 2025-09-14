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
import copy
from threading import Lock
from entity.result import Result
import pandas as pd
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
        self.dev = functions.harmony_functions
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
    def deduplication_history(self,user_input,tool_name,chat_history_id):
        if "map_tables_fields" in tool_name:
            # 当是获取映射的functions时，这里要判断一下这个函数返回的映射表是否有跟之前的某些表重复了
            # 如果这里不去掉重复的话那么 重复多次的将它传给模型，显存会暴增，毕竟一个数据表的字段可能会有很多，包括数据类型和注释
            # 而且虽然在system prompt中已经提示了模型已存在的映射表不要再次获取，但是模型有时候还是会重复获取
            # 这块或许主要还是要模型的意图能力
            # 代码上还是要去一下重
            functions_result=Result(**json.loads(user_input))
            if functions_result.data is not None:
                for i,a in enumerate(self.messages_history[chat_history_id]):
                    if a["role"] == "tool" and "map_tables_fields" in a["tool_name"]:
                        a_result=Result(**json.loads(a["content"]))
                        new_ard=copy.deepcopy(a_result.data)
                        need_replace = False
                        need_pop_indexs=[]
                        for index,ad in enumerate(a_result.data):
                            ad_find_list=list(filter(lambda x: x["table_real_name"] == ad["table_real_name"],functions_result.data))
                            if len(ad_find_list)>0:
                                need_replace=True
                                logging.info("数据表{}在历史记录中已经获取过映射，清空该表的历史记录".format(ad["table_real_name"]))
                                need_pop_indexs.append(index)
                            
                        if need_replace:
                            a_result.data=[v for i, v in enumerate(a_result.data) if i not in need_pop_indexs]
                            a["content"]=a_result.to_json()
                            logging.info("修改后的记录：{}".format(a["content"]))
                            self.messages_history[chat_history_id][i]=a
    def predict(self,chat_history_id:str, user_input:str,max_new_tokens=2048,temperature=0.05,ds_collection_info:DsCollectionInfo =None,tool_name :str=None,retry_num:int=4,stage_run_status:bool=True):
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.make_sys(current_date)
        if chat_history_id not in self.messages_history:
            self.messages_history[chat_history_id]=[]
        stage_index = 1 
        error_obj_count={}
        while True:
            if tool_name is not None:
                self.deduplication_history(user_input,tool_name,chat_history_id)
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
            last_output_div=False
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
                            last_output_div=True
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
            yield "\n"
            try:
                validate_flag = None
                commentary_content_json = None
                if commentary_name is not None and commentary_content is not None:
                    validate_flag,commentary_content_json=get_commentary_tool(commentary_name,commentary_content)
                if validate_flag is not None and not validate_flag and commentary_name is not None and commentary_content is not None:
                    logging.info("阶段{}解析functions的参数失败{}".format(stage_index,commentary_content_json))
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
                    result_info = eval(method+"(parameters,ds_collection_info)")
                    yield f"\n阶段{stage_index}执行完成..."
                    logging.info("阶段{}函数{}执行完成".format(stage_index,method))
                    yield f"\n阶段{stage_index}函数执行结果:\n"
                    if result_info.last_stage:
                        yield result_info.data
                        break
                        
                    user_input=result_info.to_json()
                    for u in list(user_input):
                        yield u
                    tool_name = method
                    stage_run_status=True
            except functions.FunctionsException as e:
                logging.error("阶段{}执行functions{}异常:{}".format(stage_index,commentary_name,traceback.format_exc()))
                # 将当前function执行失败的信息当作user_input然后采用tool这个角色再次调用模型告诉模型functions调用失败，失败原因
                stage_run_status=False
                tool_name = commentary_name
                user_input=Result(str(e)).to_json()
                if commentary_name not in error_obj_count:
                    error_obj_count[commentary_name]=1
                if error_obj_count[commentary_name] > retry_num:
                    yield f"\n执行functions{commentary_name}异常次数已达{error_obj_count[commentary_name]}次，停止输出\n"
                    break
                error_obj_count[commentary_name] += 1 
                yield f"\n阶段{stage_index}执行functions{commentary_name}异常:{str(e)}\n"
            except BaseException as e:
                logging.error(traceback.format_exc())
                if stage_index > retry_num:
                    yield f"\n系统异常次数已达{stage_index}次，停止输出\n"
                    break
                stage_run_status=False
                yield f"\n阶段{stage_index}系统出现异常，开始重试\n"
            finally:
                stage_index = stage_index+1
                if not last_output_div:
                    yield f"\n</div>\n"
                    yield f"\n</details>\n"
    def clear_chat_history(self,chat_history_id:str):
        self.messages_history.pop(chat_history_id, None) 
    def system_prompt(self):
        system_prompt="""
        你是数据库查询智能体助手  
        必须严格遵循以下协议：  
        1、分析用户问题，提取涉及的数据表，数据表名必须使用中文描述，这个中文描述可以从用户问题中提取，也可以合理猜测，但是坚决不能说不知道表结构而停止流程。最终调用 functions.map_tables_fields 传入你提取到的中文表名进行获取映射表。
        2、当获取到映射表后，请认真分析映射表，如果不满足用户当前的问题，则需要重新执行第一点获取缺失的映射表，不得胡编乱造sql语句。如果满足则直接调用functions.query_table 生成sql的规则是：
           - col_memo中可以帮助做字典类型查询过滤或格式化返回
           - SQL 中表和字段必须直接使用functions.map_tables_fields中的table_real_name和col_name，坚决不能改变它的值；
           - SQL 语句必须为分页查询，每页 5 条，查询第一页。  
           - SQL 语句不能包含换行符。  
           - sql 语句中不能用任何的关键字做别名，比如user、name、key等，必须要使用单个英文字母加单个数字的方式，如from t_user as u1
           - SELECT 中每个字段都要加 as 别名，别名使用用户问题中要求返回的中文名。 
           - 最终调用 functions.query_table，由系统执行 SQL 语句。
        3、functions.query_table该函数可能返回 SQL 执行失败的异常信息：  
           - 如果报错涉及语法错误，请对照报错信息检查并修正 SQL，然后重新执行第2点。
           - 如果报错是无法连接数据库，请你直接在final通道输出“无法连接数据库”，并结束流程。
        4、当用户发出新的查询请求时：  
           - 如果新请求中涉及的数据表或字段不在现有映射表中，必须要分析请求中新的数据表中文描述，重新执行第1点。  
           - 如果所需数据表和字段已经存在映射，则直接基于现有映射继续执行第2点。  
        5、请你注意，不管用户发出什么查询请求，你都绝对不能自己模拟数据出来返回给用户，这是非常严重的后果！必须要通过调用functions.query_table 返回数据之后你才能渲染出来输出结果。
        6、如果用户的问题涉及以下行为：新增数据、修改数据、删除数据、新增数据表、删除数据表、修改数据表，则在 final 通道直接输出：  
           `{"code":"error","msg":"对不起，我不能执行新增修改删除数据行为"}` 
        """
  
        return system_prompt
