# Websearch Agent — evaluación de Tavily como `search_tool`

Agente LangGraph (Claude Haiku 4.5) con búsqueda web vía Tavily. Se evalúa si Tavily sirve para un chatbot de consultas web y se prueban mejoras de coste y relevancia.

## Estructura
```
src/metrics.py                      # tokens y coste a partir de los mensajes (sin dependencias de LangChain)
src/websearch_agent.py              # agente baseline y optimizado, tool con solo `query`, jueces Pydantic
notebooks/evaluacion_tavily.ipynb   # planteamiento, pasos 1-3, resultados y conclusiones (ejecutado)
tests/test_metrics.py               # valida el método diferencial con datos reales
POST.md                             # borrador del post
```

## Ejecución
```bash
uv add pandas langchain langchain-anthropic langchain-tavily langgraph python-dotenv   # o pip install
# .env: ANTHROPIC_API_KEY=...  TAVILY_API_KEY=...
python tests/test_metrics.py
jupyter lab notebooks/evaluacion_tavily.ipynb     # ejecutar de arriba abajo
```

## Método
- **Tokens de un `ToolMessage`:** no tiene `usage_metadata`; se factura dentro del `input_tokens` de la llamada siguiente: `≈ input_B − input_A − output_A`. Verificado contra un recuento exacto (1 549 vs 1 537).
- **Consultas:** 8 (conocimiento estable, trampa pre-cutoff, actualidad, dato volátil, aritmética, nicho). Una ejecución por variante, `temperature=0`.
- **Variantes:** A baseline · B tool con solo `query` + resultado compacto + fecha y cutoff en el prompt · C = B + `min_score` · D = B + juez Pydantic · E = caching.

## Resultados
| Variante | Coste (8 consultas) | Ahorro vs A | Búsqueda necesaria hecha |
|---|---|---|---|
| A baseline | 0,0447 $ | — | 3/5 |
| B | 0,0295 $ | −34 % | 5/5 |
| C | 0,0251 $ | −44 % | 5/5 |
| D | 0,0401 $ | −10 % | 5/5 |

- ≈72 % del input de una consulta con búsqueda es sobrecarga fija (system prompt + schemas de tools, ≈2,3k tokens por llamada); los ToolMessage son ≈24–27 %. El schema de Tavily solo pesa ≈1 673 tokens.
- El `score` es un filtro grueso: <0,3 irrelevante; con 0,5 se perdió el único resultado útil de una consulta de nicho.
- El juez LLM reduce a la mitad el ToolMessage, pero su salida cuesta casi lo que ahorra.
- Sin `published_date` (`topic="general"`) las fuentes se contradicen en datos "de ahora" (Bitcoin).
- Prompt caching: sin efecto (prefijo menor que el mínimo cacheable).

Detalle y tablas en el notebook (§4–§5).

## Límites
8 consultas y una ejecución por variante; los resultados de Tavily cambian entre ejecuciones; el mismo modelo es agente y juez; B cambia varias cosas a la vez (ablación pendiente); la columna `OK` cuenta `multiply` como "búsqueda" (defecto de la métrica); la prueba de posición de los snippets en la página no fue concluyente.

## Pendiente
Ablación de B; suelo de score 0,3; juez mínimo y barato; `topic="news"` / `time_range`; repetir ×3 antes de publicar cifras definitivas.
