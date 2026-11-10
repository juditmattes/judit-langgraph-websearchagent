import sys, types
sys.path.insert(0, "src")
from metrics import summarize, tool_message_tokens

M = types.SimpleNamespace
# Datos reales del notebook agent_haiko_ntb (consulta del Barça)
msgs = [
    M(type="human", content="q"),
    M(type="ai", usage_metadata={"input_tokens": 2431, "output_tokens": 98}, tool_calls=[{"id": "t1"}], content=""),
    M(type="tool", tool_call_id="t1", content="x" * 5000),
    M(type="ai", usage_metadata={"input_tokens": 4523, "output_tokens": 351}, tool_calls=[], content="r"),
]
s = summarize(msgs)
assert s["input"] == 6954 and s["output"] == 449 and s["used_tool"]
assert tool_message_tokens(msgs)[0]["tokens_est"] == 1994
assert s["tool_decision_in"] == 2431 and s["tool_decision_out"] == 98
print("OK", s)
