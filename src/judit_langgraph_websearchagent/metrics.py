"""Métricas de tokens/coste sin dependencias de LangChain (duck typing sobre los mensajes).

Idea clave: un ToolMessage NO tiene usage_metadata. Sus tokens no se "facturan"
en el mensaje, sino como parte del input de la SIGUIENTE llamada al LLM.
"""
from __future__ import annotations

# USD por millón de tokens. Verificar en la página de precios de Anthropic.
PRICES = {"claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0}}
CACHE_READ_MULT, CACHE_WRITE_MULT = 0.1, 1.25


def _type(m):
    return getattr(m, "type", None)


def _usage(m) -> dict:
    u = getattr(m, "usage_metadata", None) or {}
    d = u.get("input_token_details") or {}
    return {
        "input": u.get("input_tokens", 0),
        "output": u.get("output_tokens", 0),
        "cache_read": d.get("cache_read", 0),
        "cache_creation": d.get("cache_creation", 0),
    }


def llm_calls(messages) -> list[dict]:
    """Una fila por llamada al LLM (AIMessage con usage_metadata)."""
    return [
        {"idx": i, **_usage(m), "n_tool_calls": len(getattr(m, "tool_calls", None) or [])}
        for i, m in enumerate(messages)
        if _type(m) == "ai" and getattr(m, "usage_metadata", None)
    ]


def tool_message_tokens(messages) -> list[dict]:
    """Estimación DIFERENCIAL de los tokens de cada ToolMessage.

    Llamada A (decide buscar):   input_A, output_A  (output_A = bloque tool_use)
    Llamada B (siguiente):       input_B = input_A + output_A + tokens(ToolMessages) + ~overhead
    =>  tokens(ToolMessages) ≈ input_B - input_A - output_A
    Si hay varios ToolMessage entre A y B, el total se reparte por nº de caracteres.
    Error esperado: unas pocas decenas de tokens (formato de bloques tool_use/tool_result).
    """
    ai_idx = [r["idx"] for r in llm_calls(messages)]
    out = []
    for a, b in zip(ai_idx, ai_idx[1:]):
        tms = [m for m in messages[a + 1:b] if _type(m) == "tool"]
        if not tms:
            continue
        ua, ub = _usage(messages[a]), _usage(messages[b])
        total = ub["input"] - ua["input"] - ua["output"]
        chars = [len(str(m.content)) for m in tms]
        s = sum(chars) or 1
        for m, c in zip(tms, chars):
            out.append({"tool_call_id": m.tool_call_id, "chars": c,
                        "tokens_est": round(total * c / s), "group_total": total})
    return out


def cost_usd(inp, out, cache_read=0, cache_creation=0, model="claude-haiku-4-5-20251001") -> float:
    p = PRICES[model]
    fresh = inp - cache_read - cache_creation
    return (fresh * p["in"] + cache_read * p["in"] * CACHE_READ_MULT
            + cache_creation * p["in"] * CACHE_WRITE_MULT + out * p["out"]) / 1e6


def summarize(messages, extra_in=0, extra_out=0, model="claude-haiku-4-5-20251001") -> dict:
    """Resumen de una ejecución. extra_* = tokens de llamadas auxiliares (p. ej. juez de relevancia)."""
    calls = llm_calls(messages)
    tm = tool_message_tokens(messages)
    inp = sum(c["input"] for c in calls); out = sum(c["output"] for c in calls)
    cr = sum(c["cache_read"] for c in calls); cw = sum(c["cache_creation"] for c in calls)
    decision = [c for c in calls if c["n_tool_calls"]]
    return {
        "llm_calls": len(calls),
        "used_tool": any(c["n_tool_calls"] for c in calls),
        "input": inp + extra_in, "output": out + extra_out,
        "cache_read": cr, "cache_creation": cw,
        "toolmsg_tokens": sum(t["tokens_est"] for t in tm),
        # lo que el script original llamaba "Tokens en Tool Call": coste de DECIDIR buscar, no del resultado
        "tool_decision_in": sum(c["input"] for c in decision),
        "tool_decision_out": sum(c["output"] for c in decision),
        "cost_usd": cost_usd(inp + extra_in, out + extra_out, cr, cw, model),
    }


def count_input_tokens(model, messages) -> int:
    """Tokens exactos de input: invoca con max_tokens=1 y lee usage_metadata (coste ≈ 0)."""
    r = model.invoke(messages, max_tokens=1)
    return r.usage_metadata["input_tokens"]


def exact_text_tokens(model, text: str) -> int:
    """Tokens exactos de un texto = input(texto) - input('.') + 1 (resta el envoltorio del mensaje)."""
    from langchain.messages import HumanMessage
    return count_input_tokens(model, [HumanMessage(content=text)]) - count_input_tokens(
        model, [HumanMessage(content=".")]) + 1
