"""Agente LangGraph con búsqueda web (Tavily) y herramienta personalizada.

Contiene:
    1. create_agent(...)  -> construye y compila el agente con TavilySearch.
    2. run_agent(...)     -> ejecuta el agente con una pregunta y maneja el estado.
"""

import getpass
import json
import operator
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Estado compartido del agente
# ---------------------------------------------------------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ---------------------------------------------------------------------------
# Herramienta personalizada de ejemplo
# ---------------------------------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """Suma dos números enteros `a` y `b`.

    Args:
        a: Primer número entero.
        b: Segundo número entero.
    """
    return a + b


def _ensure_api_keys() -> None:
    """Carga variables del archivo .env y solicita las claves que no estén configuradas."""
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Anthropic API key: ")
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key: ")


# ---------------------------------------------------------------------------
# 1. Función para crear el agente
# ---------------------------------------------------------------------------
def create_agent(
    model_name: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.0,
    system_prompt: str = "Eres un asistente útil e inteligente. Usa tus herramientas cuando sea necesario.",
    max_llm_calls: int = 10,
    # ---- Parámetros de búsqueda Tavily ----
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
):
    """Crea y compila el agente LangGraph.

    Args:
        model_name: Nombre del modelo a utilizar.
        temperature: Temperatura del modelo para variar respuestas.
        system_prompt: Instrucciones base para el sistema.
        max_llm_calls: Límite máximo de iteraciones con el LLM.
        max_results: Cantidad máxima de resultados recuperados en búsquedas.
        topic: Categoría o ámbito de la búsqueda ("general", "news", "finance").

    Returns:
        El grafo compilado (agente) listo para ejecutar `.invoke(...)`.
    """
    _ensure_api_keys()

    model = init_chat_model(model_name, temperature=temperature)

    # Configuración de la herramienta de búsqueda Tavily
    search_tool = TavilySearch(
        max_results=max_results,
        topic=topic,
    )

    tools = [add, search_tool]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # --- Nodo LLM: Determina la respuesta o si requiere ejecutar una herramienta ---
    def llm_call(state: dict):
        response = model_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + state["messages"]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # --- Nodo Herramientas: Ejecuta las funciones requeridas ---
    def tool_node(state: dict):
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            if not isinstance(observation, str):
                observation = json.dumps(observation, ensure_ascii=False, default=str)
            results.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    # --- Lógica condicional de flujo ---
    def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
        if state.get("llm_calls", 0) >= max_llm_calls:
            return END
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    # Construcción del grafo de estados
    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")

    return builder.compile()


# ---------------------------------------------------------------------------
# 2. Función para ejecutar el agente
# ---------------------------------------------------------------------------
def run_agent(agent, question: str, verbose: bool = True) -> str:
    """Ejecuta el agente con una consulta y retorna la respuesta final.

    Args:
        agent: Instancia del agente creada con create_agent().
        question: Pregunta o instrucción enviada por el usuario.
        verbose: Si es True, imprime la traza completa de mensajes.
    """
    state = agent.invoke({"messages": [HumanMessage(content=question)]})

    if verbose:
        for m in state["messages"]:
            m.pretty_print()

    final = state["messages"][-1].content
    if isinstance(final, list):
        final = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in final
        )
    return final


# ---------------------------------------------------------------------------
# Ejemplo de uso ejecutable directamente
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = create_agent(max_results=5, topic="general")
    answer = run_agent(agent, "¿Cuáles son los eventos tecnológicos más importantes de este año?")
    print("\nRespuesta final:\n", answer)