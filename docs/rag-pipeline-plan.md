# GovGrant AI — Plan del Pipeline RAG

**Estado:** Decisiones de arquitectura cerradas  
**Stack de datos:** LlamaIndex  
**Orquestación (posterior):** LangGraph  
**Vector store:** Qdrant  
**Fuentes:** documentos del applicant + [SBIR.gov Topics](https://www.sbir.gov/topics)

---

## 1. Objetivo

Construir primero un **RAG de producción** multimodal e híbrido (vector + BM25 + RRF), con parsers **aislados por modalidad**, consulta a corpus privado y a topics oficiales de SBIR, citas obligatorias y disclaimer de verificación en el sitio de la agency.

LangGraph se integra **después** de que existan tools de retrieve estables y citables.

---

## 2. Decisiones cerradas (Sección 7)

| # | Tema | Decisión |
|---|------|----------|
| 1 | SBIR Topics | **API oficial + API key** + cache indexado en Qdrant/store estructurado. Scrape HTML de `/topics` solo como fallback de emergencia. |
| 2 | Tablas | **Pipeline completo dual**: (a) texto/Markdown para hybrid RAG, (b) store tabular/estructurado para métricas y consultas numéricas. |
| 3 | Imágenes / gráficos | **Incluido en el plan** (OCR + caption estructurado con vision). No se pospone fuera del roadmap RAG. |
| 4 | Vector store | **Qdrant** (filtros de metadata nativos, multi-colección, hybrid-friendly). |
| 5 | Disclaimer topics | **Sí.** Toda respuesta basada en topics debe citar `https://www.sbir.gov/topics/{id}` y avisar que hay que verificar la solicitation oficial de la agency. |

---

## 3. Principios de diseño

1. **Separar por fuente y por modalidad** en ingesta; unificar en contrato de `Node` + metadata.
2. **Hybrid search por defecto** (embeddings + BM25 + RRF); no solo vectorial.
3. **Filtros de seguridad antes del retrieve** (`tenant_id`, permisos, vigencia).
4. **Citas obligatorias** en toda respuesta grounded.
5. **Re-procesado por lane**: fallar una tabla o una figura no tumba el resto del documento.
6. **Frescura de SBIR**: sync programado + revalidación puntual en queries críticas.
7. **RAG primero, agente después**: LangGraph consume tools; no define el parseo.

---

## 4. Arquitectura de alto nivel

```
                         ┌──────────────────────────────┐
          Query ────────►│  Query Router (LlamaIndex)   │
                         │  doc_qa | topic | cross | sql │
                         └──────────────┬───────────────┘
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
         Source A: User Docs     Source B: SBIR Topics    Source C: Tabular
         (tenant-privado)        (público, API+cache)     (métricas/tablas)
                │                       │                       │
     parsers por modalidad      connector API + sync      table extractors
                │                       │                       │
                ▼                       ▼                       ▼
         Qdrant + BM25            Qdrant + Postgres/JSON    SQL / structured
         (hybrid)                 + filtros agency/status    index
                │                       │                       │
                └───────────────────────┼───────────────────────┘
                                        ▼
                              RRF / multi-source fusion
                                        ▼
                              Re-rank → Context pack + citas
                                        ▼
                              Answer + disclaimer (si SBIR)
```

---

## 5. Fuentes de conocimiento

### 5.1 Source A — Corpus del applicant (privado)

| Tipo | Ejemplos |
|------|----------|
| Propuestas / drafts | Technical narrative, abstract, SOW |
| Compliance / policy adjuntos | FOA PDF, eligibility extracts |
| Presupuestos | Budget tables, cost volumes (XLSX/PDF) |
| Forms | Campos tipo SF-424, key-value |
| Figuras | Diagramas, charts de performance |

**Scope:** filtrado siempre por `tenant_id` (+ `doc_id`, `version` cuando aplique).

### 5.2 Source B — SBIR.gov Topics (público)

| Aspecto | Detalle |
|---------|---------|
| UI de referencia | https://www.sbir.gov/topics |
| Ingesta primaria | **Solicitation / Topic API** con **API key** |
| Persistencia | Store estructurado (status, fechas, agency, phase) + **Qdrant** hybrid index |
| Consulta | Hybrid semantic/keyword + filtros exactos (open/closed, agency, phase, year) |
| Live path | Re-fetch por `topic_id` en respuestas de elegibilidad/fechas críticas |
| Fallback | Cache local si API en maintenance; marcar `stale=true` |
| Cita | `https://www.sbir.gov/topics/{id}` + disclaimer de agency site |

**Nota de producto:** el propio SBIR.gov indica que los topics listados son copias y no siempre la versión más actual. El disclaimer no es cosmético: es parte del contrato de respuesta.

### 5.3 Source C — Tabular / métricas

Derivado de tablas de Source A (y, si aplica, campos numéricos de topics).

| Capa | Uso |
|------|-----|
| RAG textual de tablas | “¿qué dice la fila de indirect costs?” |
| Structured / SQL store | “suma de personal”, “% subcontract”, comparaciones numéricas |

---

## 6. Contrato común de nodos

Todo parser produce el mismo shape lógico:

```text
Node
├── text | structured_payload
└── metadata
    ├── source_type: user_doc | sbir_topic | policy
    ├── modality: prose | table | figure | chart | form
    ├── tenant_id, doc_id, version
    ├── page, section_path, parent_id
    ├── agency, topic_id, status, open_date, close_date   # SBIR
    ├── effective_from, effective_to
    ├── parse_confidence, parser_name, lane
    ├── citation_uri   # file://... o https://www.sbir.gov/topics/{id}
    └── stale: bool    # cache SBIR desactualizado / API down
```

**Qdrant:** estos campos se mapean a payload filtrable (`tenant_id`, `modality`, `source_type`, `agency`, `status`, etc.).

---

## 7. Lanes de parsing (aislados, LlamaIndex)

Cada lane es un módulo con la misma interface conceptual:

```text
BaseModalityParser.parse(input) -> list[Node]
```

### 7.1 Lane Prose / layout

| | |
|--|--|
| **Inputs** | PDF texto, DOCX, Markdown export |
| **Stack** | LlamaParse (layout → Markdown), HierarchicalNodeParser |
| **Salida** | Child chunks + parent page/section |
| **Retrieve** | Hybrid en Qdrant + BM25; ParentNodeRetriever al generar contexto |

### 7.2 Lane Tables (completo dual)

| | |
|--|--|
| **Inputs** | Tablas en PDF, XLSX, CSV, HTML tables |
| **Stack** | LlamaParse table mode + normalización `pandas` |
| **Salida A (RAG)** | Filas/grupos como texto Markdown + `modality=table` en Qdrant/BM25 |
| **Salida B (estructurado)** | Schema tabular (columnas tipadas) → SQLStructStore / DB relacional |
| **Query** | Router decide: semántica sobre tabla vs SQL/aggregate |

**Reglas:**

- Preservar headers y unidades en metadata.
- No aplanar presupuestos multi-hoja sin `sheet_name` + `table_id`.
- Tablas de budget: marcar `domain=budget` para routing prioritario.

### 7.3 Lane Figures / images

| | |
|--|--|
| **Inputs** | PNG/JPG embebidas, páginas escaneadas, diagrams |
| **Stack** | OCR + vision caption estructurado (vía LlamaParse multimodal y/o modelo vision) |
| **Salida** | `ocr_text` + `caption` + `figure_id` + link a página + imagen ref |
| **Retrieve** | Texto derivado indexado en hybrid; imagen disponible para citas/UI |

### 7.4 Lane Charts / gráficos

| | |
|--|--|
| **Inputs** | Plots, bar/line charts en PDF o imagen |
| **Stack** | Vision: ejes, series, valores legibles, unidades |
| **Salida** | Payload estructurado (`series`, `units`, `notes`) + texto serializado para BM25/vector |
| **Cuidado** | No inventar valores no legibles; `parse_confidence` bajo → guardrail |

### 7.5 Lane Forms / key-value

| | |
|--|--|
| **Inputs** | Fillable PDF, formularios fijos |
| **Stack** | Field extraction campo→valor |
| **Salida** | Pares con alta prioridad BM25 (`field_name` exacto) |

### 7.6 Lane SBIR Topic (connector, no file)

| | |
|--|--|
| **Inputs** | Respuesta API de solicitations/topics |
| **Stack** | `SBIRTopicConnector` (auth API key) → normalizer → Documents |
| **Salida** | Nodes de título, descripción, requirements, fechas, tags |
| **Upsert** | Store estructurado + colección Qdrant `sbir_topics` |

---

## 8. Indexación y retrieve (Qdrant + hybrid)

### 8.1 Colecciones Qdrant (propuesta)

| Colección | Contenido |
|-----------|-----------|
| `user_docs` | Prose, forms, figure/chart text (tenant-scoped) |
| `user_tables_rag` | Representación textual de tablas (tenant-scoped) |
| `sbir_topics` | Topics/solicitations públicos indexados |

> BM25 puede vivir en LlamaIndex (in-memory/disk por corpus) o en un índice auxiliar; debe usar tokenizador que preserve códigos (`SF-424`, `2 CFR 200`, topic codes).

### 8.2 Pipeline de indexación post-parse

1. Enrichment de metadata canónica.
2. Hierarchical split (prose) / row-group split (tables) / figure nodes.
3. Embeddings → Qdrant (payload = metadata filtrable).
4. Index BM25 paralelo sobre el mismo corpus de texto.
5. Registro de `doc_version` para re-ingest incremental.

### 8.3 Query-time hybrid

```text
Query
  → ExactMatchFilter / payload filters (tenant, agency, status, modality…)
  → Vector retriever (Qdrant)
  → BM25 retriever
  → QueryFusionRetriever (RRF)
  → (opcional) re-ranker Cohere/BGE → top 5–8
  → Parent expansion si child hit
  → Context pack con citation_uri
```

### 8.4 Multi-source fusion

| Intent | Fuentes |
|--------|---------|
| `doc_qa` | `user_docs` (+ tables rag) |
| `topic_search` | `sbir_topics` + structured filters |
| `cross_check` | user + sbir en paralelo; respuesta side-by-side con citas de ambos |
| `table_metric` | SQL / structured store (Source C) |
| `figure_qa` | nodes `modality in (figure, chart)` |

---

## 9. SBIR Topics Connector — plan operativo

### 9.1 Componentes

| Componente | Responsabilidad |
|------------|-----------------|
| `SBIRAuth` | Manejo de API key (env/secrets; nunca en repo) |
| `SBIRTopicClient` | Llamadas a Solicitation/Topic API |
| `SBIRNormalizer` | Map API → `TopicDocument` canónico |
| `SBIRStructuredStore` | Persistencia status/fechas/agency/phase |
| `SBIRIndexer` | Upsert a Qdrant `sbir_topics` + BM25 |
| `SBIRSyncJob` | Cron: open topics (+ closed recientes según política) |
| `SBIRQueryService` | Hybrid search + filters + `get_by_id` + live revalidate |

### 9.2 Flujo de sync

```text
Cron / manual trigger
  → API pull (open=1, paginado; closed según ventana)
  → normalize + dedupe por topic_id / solicitation_number
  → upsert structured store
  → re-embed solo nodos changed (hash de contenido)
  → Qdrant upsert
  → log: counts, errors, last_sync_at
```

### 9.3 Flujo de query

```text
search_sbir_topics(query, filters)
  → hybrid retrieve on sbir_topics
  → attach structured fields (status, dates)
  → if critical_freshness: GET topic by id from API
  → return hits + citation URLs + stale flag
```

### 9.4 Disclaimer obligatorio (respuesta)

Plantilla mínima a adjuntar cuando hay evidencia SBIR:

> Los topics listados en SBIR.gov pueden ser copias de las solicitations de cada agency y no necesariamente la versión más actual. Verifique siempre la solicitation oficial en el sitio de la agency y el topic en https://www.sbir.gov/topics/{id}.

---

## 10. Query Router (capa LlamaIndex; luego tool en LangGraph)

| Intent | Señales típicas | Backend |
|--------|-----------------|---------|
| Document Q&A | “en mi proposal…”, “sección 3…” | Hybrid user_docs |
| Topic discovery | “topics abiertos DoD…”, “MDA battery…” | SBIR hybrid + filters |
| Cross-check | “¿mi abstract encaja con topic X?” | Parallel user + sbir |
| Budget / métrica | “% subcontract”, “total direct costs” | Structured/SQL tables |
| Figure/chart | “el gráfico de performance…” | modality figure/chart |
| Policy lookup | códigos, CFR, eligibility text | Hybrid + BM25 boost |

**Guardrail de evidencia (pre-LangGraph, en el service):**

- Si top score / re-rank bajo o sin hits tras filtros → no generar claim factual; pedir aclaración o devolver “insufficient evidence”.
- Claims numéricos de tablas requieren hit en Source C o fila tabular citada.

---

## 11. Metadata de seguridad y multi-tenant

Aplicar **antes** del cálculo de similitud (Qdrant payload filter):

| Campo | Uso |
|-------|-----|
| `tenant_id` | Aislamiento estricto de user corpus |
| `permissions` / roles | Opcional por doc |
| `effective_from` / `effective_to` | No recuperar policy vencida |
| `version` | Preferir versión activa del doc |
| `source_type` | Evitar mezclar sbir público con privado sin intent |

**SBIR topics** no llevan `tenant_id` de customer; son corpus compartido de solo lectura.

---

## 12. Observabilidad y evaluación

### 12.1 Telemetría

- Trazas de ingesta por lane (latencia, fallos, `parse_confidence`).
- Retrieve: filtros aplicados, #hits vector vs BM25, RRF order, re-rank scores.
- SBIR sync: `last_sync_at`, error rate, stale responses.
- Costos de tokens (parse vision, embeddings, answer).

Herramienta alineada al stack: **Arize Phoenix / LlamaTrace** (como en `Infra.md`).

### 12.2 Set de regresión mínimo (desde el día del primer retrieve)

| Caso | Por qué |
|------|---------|
| Código exacto / topic ID | BM25 debe ganar |
| Paráfrasis de eligibility | Vector debe ganar |
| Tabla de budget multi-header | Dual table path |
| Figura con OCR ruidoso | No alucinar números |
| Topic open vs closed filter | Structured + payload |
| API SBIR down | Cache + `stale` + disclaimer |
| Cross-tenant leak test | Filter `tenant_id` |

---

## 13. Fases de implementación

| Fase | Nombre | Entregable | Criterio de “done” |
|------|--------|------------|--------------------|
| **R0** | Contratos | `DocumentMeta`, `Modality`, `SourceType`, `BaseModalityParser`, config Qdrant | Interfaces y carpetas sin lógica frágil |
| **R1** | Prose hybrid | LlamaParse prose → hierarchical → Qdrant + BM25 + RRF | Q&A citable sobre PDF de proposal de prueba |
| **R2** | Tables dual | Lane tablas → Qdrant textual **y** structured/SQL store | Pregunta semántica + agregación numérica sobre mismo budget |
| **R3** | SBIR connector | API key client + sync + `sbir_topics` + `search_topics` + disclaimer | Buscar topics open por keyword/agency con URL de cita |
| **R4** | Figures & charts | OCR + vision structured → index hybrid | Q&A sobre figura/chart con confidence y cita a página |
| **R5** | Multi-source router | Router + cross_check + guardrail evidencia | “abstract vs topic” con citas duales |
| **R6** | Hardening | Re-ranker, eval set CI, telemetría Phoenix | Scores de regresión baseline + dashboards |
| **R7** | LangGraph tools | Encapsular retrieve como tools + permisos/HITL | Agente usa RAG sin reimplementar parseo |

**Orden fijo acordado:** R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7.

---

## 14. Estructura de repo sugerida

```text
GovGrant-AI/
├── About.md
├── Infra.md
├── docs/
│   └── rag-pipeline-plan.md          # este documento
├── src/govgrant/
│   ├── rag/
│   │   ├── contracts.py              # metadata, enums, Node builders
│   │   ├── parsers/
│   │   │   ├── base.py
│   │   │   ├── prose.py
│   │   │   ├── tables.py
│   │   │   ├── figures.py
│   │   │   ├── charts.py
│   │   │   └── forms.py
│   │   ├── index/
│   │   │   ├── qdrant_store.py
│   │   │   ├── bm25.py
│   │   │   ├── hybrid.py
│   │   │   └── hierarchical.py
│   │   ├── tabular/
│   │   │   ├── schema.py
│   │   │   └── sql_store.py
│   │   ├── sbir/
│   │   │   ├── client.py             # API key auth
│   │   │   ├── normalizer.py
│   │   │   ├── sync.py
│   │   │   └── query.py
│   │   ├── router/
│   │   │   └── query_router.py
│   │   └── service.py                # fachada HybridRAGService
│   └── agent/                        # LangGraph (fase R7)
└── tests/
    ├── rag/
    └── fixtures/                     # PDFs, XLSX, sample topics JSON
```

---

## 15. Variables de entorno (mínimo)

| Variable | Uso |
|----------|-----|
| `QDRANT_URL` | Endpoint Qdrant |
| `QDRANT_API_KEY` | Auth Qdrant (si aplica) |
| `SBIR_API_KEY` | Auth API SBIR |
| `SBIR_API_BASE_URL` | Base URL pública SBIR API |
| `LLAMAPARSE_API_KEY` | Parseo layout/tablas/multimodal |
| `OPENAI_API_KEY` / provider keys | Embeddings + LLM + vision |
| `EMBEDDING_MODEL` | Modelo de embeddings |
| `DATABASE_URL` | Structured tables + SBIR structured store |

Secretos **nunca** en git; solo `.env` local / secret manager.

---

## 16. Criterios de aceptación globales del RAG

1. Hybrid retrieve operativo en Qdrant + BM25 + RRF.
2. Parsers de prose, tables (dual), figures/charts y forms **aislados** y re-ejecutables.
3. Multi-tenant: imposible recuperar docs de otro `tenant_id` en tests.
4. SBIR: sync con API key, search hybrid, filtros status/agency, citas URL, disclaimer.
5. Tablas: misma fuente sirve Q&A semántica y query numérica estructurada.
6. Imágenes/charts: texto derivado indexado + `parse_confidence` respetado por guardrail.
7. Toda respuesta grounded expone `citation_uri` (y disclaimer si Source B).
8. Eval set mínimo en CI o script de regresión.

---

## 17. Fuera de alcance de este plan (explícito)

- UI completa del producto.
- Workflow HITL de submission end-to-end (LangGraph R7+).
- Scraping masivo de agency sites más allá del fallback SBIR.
- Fine-tuning de modelos propios.
- Legal advice automatizado sin evidencia (el sistema debe rehusar).

---

## 18. Próximo paso inmediato

Implementar **R0 + R1**:

1. Crear skeleton `src/govgrant/rag/` con contratos y config Qdrant.
2. Lane prose + hierarchical hybrid (sin SBIR ni vision aún).
3. Fixture de un PDF SBIR-like y test de retrieve citable.

R2 (tables dual) y R3 (SBIR API key) siguen en cuanto R1 responda con citas estables.

---

## 19. Referencias internas

- Producto: `About.md`
- División LangGraph / LlamaIndex: `Infra.md`
- Topics oficiales: https://www.sbir.gov/topics
- Data/API SBIR: https://www.sbir.gov/data-resources
