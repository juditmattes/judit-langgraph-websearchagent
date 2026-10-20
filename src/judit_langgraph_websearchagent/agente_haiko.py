import os
from typing import Literal, TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# 1. Cargamos las claves secretas guardadas en el archivo .env
load_dotenv()

# 2. Elegimos el modelo de inteligencia artificial que vamos a usar (Claude)
model = init_chat_model("claude-haiku-4-5-20251001", temperature=0)

# 3. Preparamos las herramientas que el modelo puede usar
search_tool = TavilySearch(
    max_results=3,
    topic="general"
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiplica dos numeros enteros a y b."""
    return a * b

# Guardamos las herramientas en una lista y en un diccionario
tools = [multiply, search_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Conectamos las herramientas al modelo de IA
model_with_tools = model.bind_tools(tools)


# 4. Definimos el "Estado" del grafo
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# 5. NODO 1: Llama al modelo de IA con su System Prompt
def llm_call(state: dict):
    system_prompt = SystemMessage(
        content=(
            "Eres un asistente util y eficiente. Tu fecha de corte de conocimiento es julio de 2025.\n"
            "REGLAS OBLIGATORIAS PARA EL USO DE HERRAMIENTAS:\n"
            "1. Si la consulta del usuario se refiere a eventos, datos o hechos anteriores a julio de 2025, "
            "RESPONDE DIRECTAMENTE usando tu conocimiento interno. NO uses la herramienta de busqueda web.\n"
            "2. Usa la herramienta de busqueda web UNICAMENTE si la pregunta requiere informacion en tiempo real, "
            "noticias de actualidad posterior a julio de 2025, o si no tienes la respuesta en tu conocimiento previo."
        )
    )
    
    response = model_with_tools.invoke([system_prompt] + state["messages"])
    
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# 6. NODO 2: Ejecuta las herramientas solicitadas por el modelo
def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_obj = tools_by_name[tool_call["name"]]
        observation = tool_obj.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    
    return {"messages": result}


# 7. CAMINO CONDICIONAL: Decide si continuar con herramientas o terminar
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    
    return END


# 8. Construcción del Grafo de LangGraph
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compilamos el agente listo para importar directamente en el notebook
agent = agent_builder.compile()


# FUNCIONES AUXILIARES (Idénticas a agent2.py para mantener compatibilidad)

def analyze_token_usage(messages):
    """Cuenta los tokens consumidos en total y los usados especificamente en las herramientas."""
    total_input = 0
    total_output = 0
    tool_input_tokens = 0
    tool_output_tokens = 0

    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            inp = msg.usage_metadata.get("input_tokens", 0)
            out = msg.usage_metadata.get("output_tokens", 0)
            
            total_input += inp
            total_output += out
            
            if getattr(msg, "tool_calls", None):
                tool_input_tokens += inp
                tool_output_tokens += out

    print("\n--- CONSUMO DE TOKENS ---")
    print(f"- Input Tokens Totales:  {total_input}")
    print(f"- Output Tokens Totales: {total_output}")
    print(f"- Total Global:          {total_input + total_output}")
    
    if tool_input_tokens > 0:
        print(f"Tokens consumidos en Tool Call: Input={tool_input_tokens}, Output={tool_output_tokens}")
    else:
        print("Tokens consumidos en Tool Call: 0 (No se requirio busqueda)")


def inspect_tool_messages(messages):
    """Busca y muestra en pantalla lo que devolvio la busqueda de internet de forma clara."""
    tool_found = False
    print("\n--- INSPECCION DE TOOL MESSAGES ---")
    
    for msg in messages:
        if isinstance(msg, ToolMessage) or msg.type == "tool":
            tool_found = True
            print(f"> ID de llamada: {msg.tool_call_id}")
            print(f"> Contenido devuelto por la herramienta:\n{msg.content}\n")
            print("-" * 50)
    
    if not tool_found:
        print("No se generaron ToolMessages en esta ejecucion.")