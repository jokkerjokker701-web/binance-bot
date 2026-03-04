import json
from datetime import datetime

def log(event: dict):
    event["_ts"] = datetime.utcnow().isoformat()
    print(json.dumps(event, ensure_ascii=False))