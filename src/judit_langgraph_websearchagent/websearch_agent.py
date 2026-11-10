"""Agente LangGraph + Tavily: versión baseline y versión optimizada, con la misma instrumentación."""
from __future__ import annotations

import json
import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MODEL = "claude-haiku-4-5-20251001"
# Según la doc de Anthropic: conocimiento fiable hasta feb-2025, datos de entrenamiento hasta jul-2025 (verificar).
DEFAULT_CUTOFF = "febrero de 2025"


# ----------------------------------------------------------------------------- prompts
LEGACY_PROMPT = (
    "Eres un asistente util y eficiente. Tu fecha de corte de conocimiento es julio de 2025.\n"
    "1. Si la consulta se refiere a eventos anteriores a julio de 2025, RESPONDE DIRECTAMENTE. NO uses la busqueda web.\n"
    "2. Usa la busqueda web UNICAMENTE para informacion en tiempo real o posterior a julio de 2025."
)


def build_system_prompt(today: date | None = None, cutoff: str = DEFAULT_CUTOFF) -> str:
    """Inyecta cutoff y fecha actual (en vez de una tool de fecha: 0 llamadas extra, 0 tokens de schema)."""
    today = today or date.today()
    return (
        f"Eres un asistente útil y conciso. Fecha de hoy: {today.isoformat()}. "
        f"Tu conocimiento fiable llega hasta {cutoff}.\n"
        "Usa web_search solo si la pregunta depende de hechos posteriores a tu cutoff o de información cambiante "
        "(noticias, 'hoy', 'actual', 'último', precios, resultados). Si puedes responder con seguridad, responde directamente.\n"
        "Con resultados de búsqueda: usa solo lo que respaldan las fuentes, cita la URL, compara las fechas de las "
        "fuentes con la de hoy y avisa si son antiguas. Si no responden a la pregunta, dilo; no inventes."
    )


# ----------------------------------------------------------------------------- resultados de Tavily
def _clean(s: str) -> str:
    return " ".join((s or "").split())


def compact_results(raw, min_score: float | None = None, max_chars: int = 500, show_score: bool = False) -> str:
    """Convierte el dict de Tavily en texto compacto: sin campos vacíos, snippet recortado, filtro por score."""
    if not isinstance(raw, dict):
        return str(raw)
    if raw.get("error"):
        return f"ERROR_BUSQUEDA: {raw['error']}"
    kept = [r for r in raw.get("results", []) if min_score is None or r.get("score", 0) >= min_score]
    if not kept:
        return ("SIN_RESULTADOS_RELEVANTES: la búsqueda no devolvió fuentes fiables. "
                "Díselo al usuario; no inventes datos.")
    lines = []
    for i, r in enumerate(kept, 1):
        head = f"[{i}] {_clean(r.get('title'))} | {r.get('url')}"
        if r.get("published_date"):
            head += f" | {r['published_date']}"
        if show_score:
            head += f" | score={r.get('score', 0):.2f}"
        lines.append(head + "\n" + _clean(r.get("content"))[:max_chars])
    return "\n\n".join(lines)


# ----------------------------------------------------------------------------- jueces (pydantic)
class ResultRelevance(BaseModel):
    index: int = Field(description="Número [i] del resultado evaluado")
    relevante: bool = Field(description="¿Contiene información útil para responder la consulta?")
    relevancia: float = Field(ge=0, le=1, description="0 = irrelevante, 1 = responde directamente")
    explicacion: str = Field(description="Una frase: por qué sí o por qué no")


class RelevanceReport(BaseModel):
    resultados: list[ResultRelevance]


class ClaimCheck(BaseModel):
    afirmacion: str
    respaldada: bool = Field(description="¿La respaldan explícitamente las fuentes aportadas?")
    evidencia: str = Field(description="URL o fragmento que la respalda; vacío si no hay")


class GroundingReport(BaseModel):
    afirmaciones: list[ClaimCheck]

    @property
    def tasa(self) -> float:
        a = self.afirmaciones
        return sum(c.respaldada for c in a) / len(a) if a else float("nan")


def judge_relevance(judge_llm, query: str, results: list[dict], snippet_chars: int = 300):
    listing = "\n".join(
        f"[{i}] {_clean(r.get('title'))} ({r.get('url')})\n{_clean(r.get('content'))[:snippet_chars]}"
        for i, r in enumerate(results, 1))
    chain = judge_llm.with_structured_output(RelevanceReport, include_raw=True)
    out = chain.invoke([
        SystemMessage(content="Evalúas si cada resultado de búsqueda sirve para responder la consulta. "
                              "Ignora texto de navegación/boilerplate. Sé estricto."),
        HumanMessage(content=f"Consulta: {query}\n\nResultados:\n{listing}")])
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    return out["parsed"], usage


def groundedness(judge_llm, answer: str, evidence: str) -> GroundingReport:
    chain = judge_llm.with_structured_output(GroundingReport)
    return chain.invoke([
        SystemMessage(content="Descompón la respuesta en afirmaciones factuales y comprueba cada una contra las fuentes. "
                              "Una afirmación NO está respaldada si solo es inferencia del autor."),
        HumanMessage(content=f"FUENTES:\n{evidence}\n\nRESPUESTA:\n{answer}")])


def make_search_tool(max_results=3, topic="general", search_depth="basic", time_range=None,
                     include_domains=None, exclude_domains=None, min_score=None, max_chars=500,
                     judge_llm=None, judge_threshold=0.5, show_score=False):
    """Tool 'slim': el LLM solo ve `query`; el resto de parámetros de Tavily queda fijado aquí.

    Devuelve (tool, stats). stats acumula tokens del juez y resultados descartados.
    """
    kw = {k: v for k, v in dict(time_range=time_range, include_domains=include_domains,
                                exclude_domains=exclude_domains).items() if v}
    raw_tool = TavilySearch(max_results=max_results, topic=topic, search_depth=search_depth, **kw)
    stats = {"calls": 0, "judge_input": 0, "judge_output": 0, "dropped": 0, "empty": 0}

    @tool
    def web_search(query: str) -> str:
        """Busca en internet información actual. Devuelve fuentes con título, URL y extracto."""
        stats["calls"] += 1
        raw = raw_tool.invoke({"query": query})
        if isinstance(raw, dict) and raw.get("results"):
            n0 = len(raw["results"])
            if judge_llm is not None:
                report, usage = judge_relevance(judge_llm, query, raw["results"])
                stats["judge_input"] += usage.get("input_tokens", 0)
                stats["judge_output"] += usage.get("output_tokens", 0)
                if report is not None:
                    ok = {r.index for r in report.resultados if r.relevante and r.relevancia >= judge_threshold}
                    raw = {**raw, "results": [r for i, r in enumerate(raw["results"], 1) if i in ok]}
            text = compact_results(raw, min_score, max_chars, show_score)
            kept = 0 if text.startswith("SIN_RESULTADOS") else text.count("\n\n[") + 1
            stats["dropped"] += n0 - kept
        else:
            text = compact_results(raw, min_score, max_chars, show_score)
        stats["empty"] += text.startswith("SIN_RESULTADOS")
        return text

    return web_search, stats


# ----------------------------------------------------------------------------- grafo
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def json_serializer(obs):
    return obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False, default=str)


def legacy_serializer(obs):  # el que usa agente_haiko.py
    return str(obs)


def _cached(msg):
    c = msg.content
    blocks = [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return msg.model_copy(update={"content": blocks})


def build_agent(tools, system_prompt, model_name=MODEL, max_llm_calls=6, serializer=json_serializer, cache=False):
    """cache=True marca breakpoints (system + último mensaje). Haiku 4.5 exige >=4096 tokens de prefijo (verificar)."""
    load_dotenv()
    model = init_chat_model(model_name, temperature=0)
    with_tools = model.bind_tools(tools)
    by_name = {t.name: t for t in tools}

    def llm_call(state):
        n = state.get("llm_calls", 0)
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        if cache:
            msgs = [_cached(msgs[0])] + msgs[1:-1] + [_cached(msgs[-1])]
        # En la última iteración permitida se quitan las tools: el modelo debe responder (evita finales vacíos).
        runner = model if n >= max_llm_calls - 1 else with_tools
        return {"messages": [runner.invoke(msgs)], "llm_calls": n + 1}

    def tool_node(state):
        out = []
        for tc in state["messages"][-1].tool_calls:
            try:
                obs = serializer(by_name[tc["name"]].invoke(tc["args"]))
            except Exception as e:  # noqa: BLE001
                obs = f"ERROR_HERRAMIENTA: {e}"
            out.append(ToolMessage(content=obs, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out}

    def should_continue(state) -> Literal["tool_node", "__end__"]:
        return "tool_node" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(MessagesState)
    g.add_node("llm_call", llm_call)
    g.add_node("tool_node", tool_node)
    g.add_edge(START, "llm_call")
    g.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    g.add_edge("tool_node", "llm_call")
    return g.compile()


def final_text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


def chat(agent, question: str, history: list | None = None):
    """Un turno. Devuelve (respuesta, historial_nuevo, mensajes_nuevos_de_este_turno)."""
    history = history or []
    state = agent.invoke({"messages": history + [HumanMessage(content=question)], "llm_calls": 0})
    return final_text(state["messages"][-1]), state["messages"], state["messages"][len(history):]
