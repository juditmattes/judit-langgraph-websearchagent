# Websearch Agent — evaluación de Tavily como `search_tool`

Agente LangGraph (Claude Haiku 4.5) con búsqueda web vía Tavily. Objetivo: evaluar si Tavily es adecuado para un chatbot de consultas web de la empresa y si se puede mejorar coste y relevancia.

## Estructura
```
src/metrics.py            # métricas de tokens/coste (sin dependencias de LangChain)
src/websearch_agent.py    # agente baseline y optimizado, tool slim, jueces Pydantic
notebooks/evaluacion_tavily.ipynb   # planteamiento, pasos 1-3, resultados, conclusiones
tests/test_metrics.py     # valida el método diferencial con datos reales (python tests/test_metrics.py)
```

## Instalación y ejecución
```bash
pip install langchain langchain-anthropic langchain-tavily langgraph python-dotenv pandas
# .env: ANTHROPIC_API_KEY=...  TAVILY_API_KEY=...
jupyter lab notebooks/evaluacion_tavily.ipynb
```

## Qué se mide
| Paso | Pregunta | Dónde |
|---|---|---|
| 1 | ¿Usa la tool cuando debe? ¿Cuánto pesan los `ToolMessage`? | §1 |
| 2 | Estructura de la respuesta, schema que ve el LLM, snippets, score, alucinación | §2 |
| 3 | Variantes B–E (tool slim, `min_score`, juez Pydantic, prompt caching) | §3 |

### Tokens de los ToolMessage (resumen)
Un `ToolMessage` no tiene `usage_metadata`; su peso se factura como parte del `input_tokens` de la **siguiente** llamada al LLM:

`tokens(ToolMessages) ≈ input_B − input_A − output_A`

(A = llamada que decide buscar, B = la siguiente). No confundir con los tokens del `AIMessage` con `tool_calls`, que miden la *decisión* de buscar. Detalle y verificación exacta en el notebook §1.1.

## Hallazgos iniciales (variante A, 3 consultas)
- Decisión tool/no-tool correcta en las 3.
- Con búsqueda: 6 954 tokens de input, de los que ≈1 994 son ToolMessages (≈29 %) y ≈4,8k (≈69 %) sobrecarga fija de system prompt + schemas de tools, reenviada en cada llamada.
- Snippets con boilerplate de navegación y páginas índice; el `score` no lo penaliza.
- El agente dice "hoy" sin conocer la fecha ni tener `published_date`.

## Cambios propuestos respecto al código original
- System prompt con fecha y cutoff en vez de una tool de fecha.
- Tool con solo `query` visible para el LLM; resto de parámetros de Tavily fijados.
- ToolMessage compacto (sin `None`/`[]`, snippets recortados, JSON/texto en vez de `str(dict)`).
- Búsqueda infructuosa → mensaje corto `SIN_RESULTADOS_RELEVANTES`.
- Última iteración permitida sin tools (evita respuestas vacías al alcanzar `max_llm_calls`).

## Estado y límites
Las celdas no se han ejecutado todavía con las claves; las cifras de §4 proceden de las ejecuciones previas. Hay que verificar: precios, mínimo de tokens para caché en Haiku 4.5, cutoff del modelo y qué parámetros de Tavily expone la versión instalada de `langchain-tavily`.
