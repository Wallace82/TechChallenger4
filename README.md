> **Tech Challenge FIAP — Pós Tech em IA para Devs — Fase 5: Guardiã AI.**
> Projeto acadêmico. **Não substitui avaliação clínica.** O classificador de risco usa **dados 100 % sintéticos**. Sem validação clínica.

As sprints PI-S01…PI-S08 estão **fechadas** (`docs/planos/`). O produto (ML + LangGraph + Gradio + Docker) está nesta branch. O que ainda é entrega humana ou git está em [Passos que faltam para finalização](#passos-que-faltam-para-finalização).

## Evolução ML (2026-09-18)

Pipeline: dados estruturados → `lib/ml` → explicabilidade → regras `SINAIS_ALARME_OBST` → workflow `risco_ml` → RAG (quando houver retriever) → síntese LLM somente-leitura (`FakeChatModel` no CPU) → Gradio (6ª aba) → auditoria `predicoes_ml`.

- Dataset: 8000 linhas, semente 42, sha256 em `artifacts/data/risco_gestacional_v1.manifest.json`.
- Vencedor por PR-AUC no teste: **regressão logística** (0,590; recall+ 0,955 no limiar 0,278 da validação). Fonte: `artifacts/metrics/comparacao.json`.
- 10 tools LangChain (a 10ª é `predizer_risco_gestacional`); 5 grafos (os 4 da Fase 3 + `risco_ml`); 6 abas Gradio.
- Demo CLI: `python scripts/run_demo.py`. UI: `python scripts/app.py` → http://localhost:7860.
- Docker `demo-cpu` usa `FakeChatModel`, **não** o Llama 3.2 3B. Log: [`docs/deploy/EXECUCAO_DOCKER.md`](docs/deploy/EXECUCAO_DOCKER.md).

## Como executar o projeto

Caminho padrão após o clone: CPU local, sem GPU e sem Hugging Face. Llama + Chroma completo é só o [pipeline Colab](#como-rodar-no-colab-llama--chroma).

### Requisitos

- Python 3.12 ou 3.13
- Git; clone da `main`
- (Opcional) Docker Desktop

### 1. Instalar

```text
git clone https://github.com/Evandrorsampaio/TechChallenger4.git
cd TechChallenger4
pip install -r requirements.txt -r requirements-ml.txt
```

Windows: `copy .env.example .env`  
Linux/macOS: `cp .env.example .env`

Não é preciso `HF_TOKEN` neste perfil.

### 2. Conferir o modelo e os testes

```text
python scripts/train.py --verificar-dataset
python -m pytest -q
```

`--verificar-dataset` valida o manifesto SHA-256 (`artifacts/data/risco_gestacional_v1.manifest.json`). Só rode `python scripts/train.py` sem flag se quiser retreinar.

### 3. Demo em quatro cenários (CLI)

```text
python scripts/run_demo.py
```

Gera `artifacts/demo/D1_sucesso.json` … `D4_degradado.json` (normal, incompleto, bypass de emergência, degradado) com `retrieved_sources` e `doc_id` do subset RAG.

### 4. Interface Gradio (Inicialização Rápida)

Você pode iniciar toda a aplicação com o script PowerShell:

```powershell
# Iniciar via Docker (padrão)
.\start.ps1

# Ou iniciar em ambiente Python local
.\start.ps1 -Mode local
```

Ou diretamente via Python:

```text
python scripts/app.py
```

Abra **http://localhost:7860**. Título: **Guardiã AI**. Seis abas; na sidebar escolha qualquer paciente (nomes Faker). CPU usa `FakeChatModel` e RAG lexical (`lib/rag_local.py`).

### 5. Docker

Executar a **aplicação completa (Gradio UI)** via Docker:

```text
docker build -t techchallenger4 .
docker run -p 7860:7860 techchallenger4
```
Acesse a aplicação no navegador em **http://localhost:7860**.

Ou utilizando **Docker Compose**:

```text
docker compose up --build
```

Executar a **demonstração em lote CLI** (4 cenários no terminal):

```text
docker run --rm techchallenger4 python scripts/run_demo.py
```

Executar a **suíte de testes** dentro do contêiner:

```text
docker run --rm techchallenger4 pytest
```

A imagem utiliza o perfil `demo-cpu` (`FakeChatModel`, scikit-learn e RAG lexical). Detalhes e logs em [`docs/deploy/EXECUCAO_DOCKER.md`](docs/deploy/EXECUCAO_DOCKER.md).

### 6. Relatório HTML / PDF

```text
python scripts/export_pdf_html.py
```

Abre `RELATORIO_TECNICO.html`; no Chrome: Imprimir → Salvar como PDF.

## Passos que faltam para finalização

O código do produto **já está nesta branch**. Falta só o que a banca pede fora do repositório ou no Git de entrega.

### Ainda humano / git

1. **PDF da banca** se o HTML não bastar: passo 6 da execução.

### Opcional (não bloqueia o enunciado)

4. Capturas da 6ª aba (quatro modos). Índice **Chroma** no Colab (`06_indexar_protocolos.ipynb`, chunk 1000/200) se for mostrar GPU. Pesos MiniLM locais. Llama + QLoRA no Colab. **`shap`** se o Python permitir. Bump de `pip-audit` com regressão.

### Fora do escopo (não afirmar o contrário)

5. Validação clínica em dados reais, SSO, criptografia at-rest.

# Guardiã AI — Saúde e Segurança da Mulher

Assistente virtual de apoio à equipe de saúde (médicos, enfermeiros, residentes, técnicos) especializado em **saúde e segurança da mulher**. Jornada principal: **triagem de gestante** (aba Risco Gestacional ML) e **relato de segurança** (aba Violência). Construído a partir de:

- **Fine-tuning QLoRA** do Llama 3.2 3B Instruct sobre 6414 pares Q&A sintéticos derivados de protocolos do Ministério da Saúde, FEBRASGO, OMS e INCA;
- **RAG** sobre 1392 chunks dos mesmos protocolos via ChromaDB + embeddings multilíngues;
- **Agente LangChain** com **10** ferramentas estruturadas (as 9 da Fase 3 + `predizer_risco_gestacional`);
- **5 fluxos LangGraph** (Triagem, Violência, Obstétrico, Prevenção, **Risco gestacional ML**);
- **UI Gradio** com **6** abas (as 5 da Fase 3 + Risco Gestacional ML).

> **Tech Challenge FIAP — Pós Tech em IA para Devs — Fase 5 (Guardiã AI).**
> Projeto acadêmico. Não substitui avaliação clínica profissional.

---

## Sumário

- [Evolução ML (2026-09-18)](#evolução-ml-2026-09-18)
- [Como executar o projeto](#como-executar-o-projeto)
- [Passos que faltam para finalização](#passos-que-faltam-para-finalização)
- [Demonstração](#demonstração)
- [Arquitetura](#arquitetura)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Stack técnico](#stack-técnico)
- [Como rodar no Colab (Llama + Chroma)](#como-rodar-no-colab-llama--chroma)
- [Pipeline em 10 notebooks](#pipeline-em-10-notebooks)
- [Categorias cobertas](#categorias-cobertas)
- [Segurança, ética e LGPD](#segurança-ética-e-lgpd)
- [Limitações conhecidas](#limitações-conhecidas)
- [Documentação adicional](#documentação-adicional)

---

## Demonstração

Capturas dos 4 fluxos LangGraph rodando ponta-a-ponta estão disponíveis em [`09_demo_workflows.ipynb`](09_demo_workflows.ipynb) (com diagramas Mermaid auto-gerados) e a UI integrada em [`08_app_gradio.ipynb`](08_app_gradio.ipynb).

JSON dos quatro modos da demo CLI: `artifacts/demo/`. Notebooks Colab: [`09_demo_workflows.ipynb`](09_demo_workflows.ipynb), [`08_app_gradio.ipynb`](08_app_gradio.ipynb).

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│           UI Gradio — Guardiã AI (6 abas; CPU ou Colab)          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
       ┌─────────────────────┴──────────────────────────┐
       │              5 LangGraph Workflows              │
       │  Triagem | Violência | Obstétrico | Prevenção   │
       │                    risco_ml                     │
       └───────┬────────────────────────────────────────┘
               │
       ┌───────▼────────────────────────────────────────┐
       │  10 Tools (LangChain StructuredTool)            │
       │  9 da Fase 3 + predizer_risco_gestacional       │
       │  buscar_protocolo (RAG local CPU ou Chroma)     │
       └─┬──────────────────────────────┬───────────────┘
         │                              │
   ┌─────▼─────┐                 ┌──────▼──────────────┐
   │ SQLite    │                 │ RAG                 │
   │ 50 pacs   │                 │ subset JSON (CPU)   │
   │ predicoes │                 │ Chroma MiniLM Colab │
   │ _ml       │                 └─────────────────────┘
   └───────────┘
         │
   ┌─────▼──────────────────────────────────────────────┐
   │ LLM: FakeChatModel (CPU/Docker) ou Llama 3.2 3B    │
   │      + QLoRA (Colab). Somente leitura dos números. │
   └────────────────────────────────────────────────────┘
```

Detalhes Fase 3: [`ARQUITETURA.md`](ARQUITETURA.md). Alvo da evolução: [`docs/arquitetura/ARQUITETURA_ALVO.md`](docs/arquitetura/ARQUITETURA_ALVO.md).

---

## Estrutura do repositório

```
fine-tuning-rag-documentos-fiap/
├── README.md                              ← você está aqui
├── ARQUITETURA.md                         ← decisões técnicas, schema, contratos
├── RELATORIO_TECNICO.md                   ← relatório executivo (visão de ~3 páginas)
├── RELATORIO_TECNICO_DETALHADO.md         ← relatório acadêmico estendido
├── .gitignore                             ← exclui .env, dados grandes, adapters
│
├── 01_extrair_protocolos.ipynb            ← PDF → JSON (PyMuPDF)
├── 02_gerar_dataset_sft.ipynb             ← chunking + geração Q&A via Llama 3.1 8B
├── 03_treinar_qlora.ipynb                 ← SFT QLoRA do Llama 3.2 3B
├── 04_avaliar_modelo.ipynb                ← base vs FT (ROUGE + heurísticas)
├── 05_gerar_dados_mock.ipynb              ← popula hospital.db
├── 06_indexar_protocolos.ipynb            ← indexa Chroma
├── 07_testar_tools_alertas.ipynb          ← sanity check das tools
├── 08_app_gradio.ipynb                    ← orquestrador UI completo
├── 09_demo_workflows.ipynb                ← demo dos 4 fluxos + Mermaid
├── 10_relatorio_utilizacao.ipynb          ← relatório gerencial: cobertura, auditoria LGPD, KPIs
│
├── scripts/                               ← train, evaluate, predict, run_demo, app
├── tests/                                 ← unit / integration / e2e / regression
├── artifacts/                             ← metrics, models (cards), demo, explainability
├── docs/                                  ← planos, ML, deploy, requisitos, demo
│
└── lib/
    ├── db.py                              ← SQLite + 7 tabelas Fase 3 + predicoes_ml
    ├── ml/                                ← schema, dataset, treino, predict, explain
    ├── mock_data.py                       ← Faker pt_BR + cenários clínicos
    ├── alertas.py                         ← regras determinísticas MS/FEBRASGO
    ├── tools.py                           ← 10 StructuredTools LangChain
    ├── llm.py                             ← load_finetuned + ChatHuggingFace
    ├── agent.py                           ← LangGraph ReAct agent
    ├── ui.py                              ← Gradio Blocks (chat + triagem)
    ├── workflows/
    │   ├── common.py                      ← llm_json, rag_search, estimar_confianca
    │   ├── triagem.py                     ← StateGraph: 7 nodes + edge condicional
    │   ├── violencia.py                   ← StateGraph: 7 nodes + matriz SINAN
    │   ├── obstetrico.py                  ← StateGraph: 7 nodes + alertas obstétricos
    │   ├── prevencao.py                   ← StateGraph: 6 nodes + agendamento
    │   └── risco_ml.py                    ← validação → regras → ML → RAG → LLM
    └── templates/                         ← modelos especializados de documentos clínicos
        ├── README.md                      ← índice + instruções de uso
        ├── laudo_mamografia_birads.md     ← laudo BI-RADS (ACR / INCA)
        ├── laudo_colposcopia_biopsia.md   ← colposcopia + AP de colo (IFCPC 2011 / INCA)
        ├── receita_terapia_hormonal.md    ← receita COC/POP/DIU/TRH com checklist FEBRASGO
        ├── acompanhamento_prenatal_puerperio.md  ← caderneta pré-natal + 2 puerperais (EPDS)
        ├── ficha_notificacao_sinan_violencia.md  ← Ficha SINAN compulsória
        └── relatorio_atendimento_violencia.md    ← relatório circunstanciado (Norma Técnica MS)
```

---

## Stack técnico

| Camada | Tecnologia | Versão |
|---|---|---|
| LLM base | meta-llama/Llama-3.2-3B-Instruct | — |
| LLM gerador (dataset) | meta-llama/Llama-3.1-8B-Instruct | — |
| Fine-tuning | QLoRA (4-bit NF4 + bfloat16) | bitsandbytes ≥ 0.45 |
| Treino | TRL SFTTrainer + PEFT | trl ≥ 0.12, peft ≥ 0.13 |
| Embeddings | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384d; `max_seq_length` 128 no Hub |
| Vector store | Chroma | persistente em Drive (ausente na imagem CPU) |
| ML tabular | scikit-learn (LogReg, RF, Dummy, regra) | `requirements-ml.txt` |
| Orquestração | LangChain + LangGraph | pins em `requirements.txt` |
| Base estruturada | SQLite (7 tabelas Fase 3 + `predicoes_ml`) | stdlib |
| UI | Gradio Blocks | 6 abas |
| Runtime ML/demo | CPU local ou Docker `demo-cpu` | sem Llama |
| Runtime LLM | Google Colab Pro (A100/L4) | Fase 3 |

---

## Como rodar no Colab (Llama + Chroma)

Atalho: abra [`COLAB_EXECUTAR.ipynb`](COLAB_EXECUTAR.ipynb) no Colab (GPU) e **Executar tudo**. Clona o repo, monta o Drive, sobe Gradio (`share=True`). Com `HF_TOKEN` + GPU carrega o Llama; senão usa `FakeChatModel`. Não substitui o treino QLoRA (`02`/`03`).

Para a demo local, use [Como executar o projeto](#como-executar-o-projeto) (CPU). O restante desta seção é o pipeline **Llama + RAG** notebook a notebook (Fase 3).

### Pré-requisitos (Fase 3 / GPU)

1. **Google Colab Pro** (ou Pro+) com GPU **A100** ou **L4**.
2. **Conta HuggingFace** com aprovação Meta para `Llama-3.1-8B-Instruct` e `Llama-3.2-3B-Instruct`.
3. **Google Drive** com a seguinte estrutura mínima:
   ```
   /MyDrive/AssistenteHospitalar/
   ├── .env                              ← HF_TOKEN=hf_xxx
   ├── lib/                              ← upload da pasta lib/ deste repo
   │   ├── db.py, tools.py, llm.py, agent.py, ui.py, ...
   │   └── workflows/
   └── files/
       └── (PDFs originais dos protocolos por categoria — opcional, se for refazer extração)
   ```

A pasta `lib/` precisa estar **diretamente em `AssistenteHospitalar/`** (não dentro de subpasta), porque os notebooks fazem `sys.path.insert(0, '/content/drive/MyDrive/AssistenteHospitalar')` e depois `from lib import ...`.

### Pipeline em 10 notebooks

Cada notebook é numerado pela ordem de execução. Saídas de um alimentam o próximo via Drive.

| # | Notebook | Saída | Tempo |
|---|---|---|---|
| 01 | [`01_extrair_protocolos.ipynb`](01_extrair_protocolos.ipynb) | `files/fontes_saude_mulher_v2.json` (39 PDFs estruturados) | 5 min |
| 02 | [`02_gerar_dataset_sft.ipynb`](02_gerar_dataset_sft.ipynb) | `files/sft/sft_train.jsonl` (5134 ex) + val/test | 6-8h em A100 |
| 03 | [`03_treinar_qlora.ipynb`](03_treinar_qlora.ipynb) | `files/finetune/llama32-3b-saude-mulher_<ts>/adapter_final/` | 2-3h em A100 |
| 04 | [`04_avaliar_modelo.ipynb`](04_avaliar_modelo.ipynb) | `eval_report.json` + `eval_sidebyside.md` | 5 min |
| 05 | [`05_gerar_dados_mock.ipynb`](05_gerar_dados_mock.ipynb) | `files/hospital.db` (50 pacientes mock) | <1 min |
| 06 | [`06_indexar_protocolos.ipynb`](06_indexar_protocolos.ipynb) | `files/chroma/` (1392 chunks indexados) | ~5 min |
| 07 | [`07_testar_tools_alertas.ipynb`](07_testar_tools_alertas.ipynb) | logs no notebook (sanity check) | 2 min |
| 08 | [`08_app_gradio.ipynb`](08_app_gradio.ipynb) | URL pública Gradio (≈72h) | 3 min |
| 09 | [`09_demo_workflows.ipynb`](09_demo_workflows.ipynb) | demos dos 4 workflows + diagramas Mermaid | 5 min |
| 10 | [`10_relatorio_utilizacao.ipynb`](10_relatorio_utilizacao.ipynb) | relatório gerencial: cobertura preventiva, auditoria LGPD, indicadores epidemiológicos | 1 min |

**Dependências:** 03 e 04 dependem de 02 (dataset). 05 e 06 são independentes e podem rodar em paralelo. 07/08/09 precisam de 05+06 (e idealmente 03+04 para o adapter).

Após o passo 08 ou 09, o link público do Gradio (via `share=True`) fica ativo por ~72h.

---

## Categorias cobertas

| Categoria | Slug interno | Sensitive | # PDFs | # Chunks |
|---|---|---|---|---|
| Ginecologia e obstetrícia | `ginecologia_obstetricia` | não | 19 | 257 |
| Câncer de mama e colo | `cancer_mama_colo` | não | 4 | 242 |
| Planejamento familiar | `planejamento_familiar` | não | 6 | 484 |
| Violência doméstica | `violencia_domestica` | **sim** | 8 | 237 |
| Saúde mental | `saude_mental` | **sim** | 2 | 172 |
| **Total** | — | — | **39** | **1392** |

Categorias `sensitive: true` recebem prompts diferenciados durante a geração do dataset (foco em encaminhamento SINAN, escuta qualificada, fluxo da rede) e ativam o **fluxo dedicado de violência** no LangGraph.

---

## Segurança, ética e LGPD

### Limites de atuação codificados

Hard-coded no system prompt do agente e nos workflows:

- **NUNCA** prescreve medicação sem validação de especialista
- **NUNCA** diagnostica condições definitivamente
- **SEMPRE** encaminha casos suspeitos de violência para profissionais qualificados (notificação SINAN compulsória)
- **SEMPRE** sugere avaliação presencial para sintomas alarmantes
- **NÃO** se dirige à paciente — é ferramenta de apoio à equipe de saúde

### Auditoria LGPD

Toda chamada à `consultar_violencia` e `registrar_violencia` grava em `log_acesso` com:
- timestamp
- usuario (identificador do profissional)
- tabela acessada
- paciente_id
- **motivo clínico obrigatório** (mínimo 5 caracteres)

### Dados pessoais

- **Mock 100% sintético** via Faker pt_BR (sem dado real)
- **CPF hash** SHA-256 (nunca CPF real)
- Tabela `registros_violencia` em schema separado

### Limites técnicos

- Não há criptografia at-rest (uso acadêmico em mock); para produção, recomenda-se Fernet em `registros_violencia.observacoes`
- Não há autenticação federada — `usuario` é mock simples; em produção viria de OAuth/SSO

---

## Limitações conhecidas

### Resultados do fine-tuning (run `llama32-3b-saude-mulher_20260524_0217`)

**Ganhos confirmados:**
- ✅ Citação de serviços da rede (SINAN, Ligue 180, CVV, CAPS) em casos sensitive saltou de ~33% (base) para ~100% (FT)
- ✅ Tom inicial mais alinhado ao público profissional (não inicia com "Lamento ter de abordar...")

**Regressões mensuráveis:**
- ❌ ROUGE-L caiu de 0.101 para 0.095 (-5.9%) — proxy fraco mas regressão real
- ❌ ~60% das respostas FT apresentam **loops repetitivos** — mitigado com `repetition_penalty=1.2` + `no_repeat_ngram_size=4` aplicados em [`lib/llm.py`](lib/llm.py)
- ❌ FT cita serviços da rede inadequadamente em contextos não-sensitive (overfitting do padrão sensitive)
- ❌ Respostas 18.6% mais longas que o base — mitigado com `max_new_tokens=256`

Análise detalhada no [`RELATORIO_TECNICO_DETALHADO.md`](RELATORIO_TECNICO_DETALHADO.md#43-análise-crítica-honesta-dos-resultados).

### Limitações estruturais

1. **Validação clínica formal pendente** — Q&A SFT e o dataset tabular de risco são sintéticos. Validação por especialistas é necessária antes de qualquer uso real.
2. **Tool calling do Llama 3.2 3B** pode ter taxa de erro maior que modelos maiores em cenários complexos. Fallback ReAct presente.
3. **RAG enviesado pela base de protocolos** — coberta MS/FEBRASGO/OMS/INCA brasileiros. Pode ter gaps em protocolos institucionais específicos.
4. **Sem feedback de profissionais reais** — métricas atuais são heurísticas + ROUGE-L, não satisfação clínica.
5. **Anonimização** dos dados reais não está implementada como pipeline — apenas demonstrada via mock. Para produção, seria necessário pipeline de PHI removal antes do fine-tuning.
6. **Tamanho amostral da avaliação pequeno** (n=30 de 640 do test set) — para conclusões estatísticas seria necessário n≥100 com bootstrap.

---

## Documentação adicional

- **[`docs/planos/00_INDICE.md`](docs/planos/00_INDICE.md)** — planos PI-S01…S08 e [relatório do ciclo](docs/planos/RELATORIO_CICLO_FINAL.md)
- **[`docs/ml/METRICAS_E_RESULTADOS.md`](docs/ml/METRICAS_E_RESULTADOS.md)** — números do teste (fonte: `artifacts/metrics/`)
- **[`docs/deploy/EXECUCAO_DOCKER.md`](docs/deploy/EXECUCAO_DOCKER.md)** / [`EXECUCAO_LOCAL.md`](docs/deploy/EXECUCAO_LOCAL.md)
- **[`docs/requisitos/CRITERIOS_DE_ACEITE.md`](docs/requisitos/CRITERIOS_DE_ACEITE.md)** — 20 critérios da evolução
- **[`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md)** — relatório executivo Fase 5 (Guardiã AI)
- **[`RELATORIO_TECNICO_DETALHADO.md`](RELATORIO_TECNICO_DETALHADO.md)** — relatório acadêmico estendido Fase 3
- **[`ARQUITETURA.md`](ARQUITETURA.md)** — decisões técnicas Fase 3

---

## Tech Challenge — Critérios de avaliação

| Critério | Onde encontrar |
|---|---|
| Precisão médica especializada | Dataset baseado em MS/FEBRASGO/OMS/INCA; RAG cita fonte |
| Segurança da paciente | Workflow de violência (notificação SINAN, rede de proteção); limites no system prompt |
| Sensibilidade ética | Prompts diferenciados para `sensitive`; tom técnico (assistente para profissional, não paciente) |
| Aplicabilidade prática | Mock realista de prontuário/exames/violência; UI Gradio funcional |
| Impacto social | Apoio a equipe de saúde da mulher (categoria com déficit de protocolos digitalizados no Brasil) |
| Conformidade regulatória | `log_acesso` (LGPD), CPF hashado, separação de dados sensíveis |
