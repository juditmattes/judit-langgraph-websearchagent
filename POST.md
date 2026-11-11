# Cuánto cuesta de verdad que un agente busque en la web (Tavily + LangGraph + Claude Haiku 4.5)

Cuando se añade una herramienta de búsqueda a un agente, la intuición es que lo caro son los resultados. Lo medimos en un agente LangGraph con Tavily y Claude Haiku 4.5, y no era así.

## 1. El coste dominante es lo que se reenvía en cada llamada
Un `ToolMessage` no tiene `usage_metadata`: sus tokens se facturan dentro del `input_tokens` de la llamada siguiente. Se estiman con `input_B − input_A − output_A` (A = la llamada que decide buscar, B = la siguiente), y lo verificamos contra un recuento exacto: 1 549 frente a 1 537 tokens.

Con ese método, en una consulta con búsqueda de 3 resultados:

- Cada llamada reenvía ≈2,3k tokens de system prompt + schemas de tools. Con dos llamadas, eso es ≈72 % del input.
- Los ToolMessage son ≈24–27 %.
- Solo el schema de Tavily pesa ≈1 673 tokens, y el LLM casi nunca necesita decidir sus argumentos.

Cuidado con una trampa habitual: sumar los tokens de los mensajes con `tool_calls` mide el coste de *decidir* buscar, no el de inyectar el resultado.

## 2. Tres cambios simples
Con 8 consultas, comparamos el agente base con una versión que:
- expone al LLM una tool con solo `query` (el resto de parámetros de Tavily queda fijado),
- compacta el resultado (sin campos vacíos, snippets de ≤500 caracteres),
- inyecta la fecha de hoy y el cutoff en el system prompt.

| | Base | Mejorada |
|---|---|---|
| Coste de las 8 consultas | 0,0447 $ | 0,0295 $ (−34 %) |
| Búsquedas necesarias realizadas | 3/5 | 5/5 |

El agente base nunca buscaba de más, pero se saltaba búsquedas necesarias (por ejemplo "¿quién ganó la última Champions?") porque no sabía qué día era.

## 3. Filtrar resultados no es gratis ni trivial
- **Por score:** el `score` de Tavily separa bien los extremos (por debajo de 0,3 todo era irrelevante), pero con un umbral de 0,5 se perdió el único resultado útil de una consulta de nicho.
- **Con un juez LLM** (schema Pydantic: relevante, relevancia, explicación) el ToolMessage se reduce a la mitad, pero el juez gasta casi lo que ahorra: solo −10 % de coste total. Lo caro es su salida, no su entrada.
- **Prompt caching:** no hizo nada; el prefijo es menor que el mínimo cacheable.

## 4. Lo que no sabemos
Son 8 consultas y una ejecución por variante. Tavily devuelve resultados distintos entre ejecuciones, el mismo modelo hace de agente y de juez, y la versión mejorada cambia varias cosas a la vez, así que no sabemos cuánto aporta cada una. También vimos que, sin fecha de publicación, las fuentes se contradicen en datos "de ahora" (dos precios de Bitcoin distintos con diez mil dólares de diferencia).

## Próximos pasos
Ablación de cada cambio, suelo de score de 0,3, juez mínimo y barato, `topic="news"` para actualidad, y repetir cada variante varias veces.

*Código y notebook: [enlace al repositorio].*
