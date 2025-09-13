import json
class Result:
    def __init__(self,msg,data=None,last_stage=False):
        self.msg=msg
        self.data=data
        self.last_stage=last_stage
    def to_json(self):
        return json.dumps({
            "msg": self.msg,
            "data": self.data
        }, ensure_ascii=False)