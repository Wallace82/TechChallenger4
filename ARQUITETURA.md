# Guardiã AI — Arquitetura do Sistema (Fase 5)

Documento consolidado de arquitetura da **Guardiã AI** (Assistente Clínico Hospitalar com Machine Learning, RAG, LLM e LangGraph).

**Público-alvo:** Equipe de saúde hospitalar (médicos, enfermeiros, residentes, técnicos). **Não** é assistente para a paciente — todas as respostas e orientações assumem o contexto de um profissional durante o atendimento clínico.

---

## 1. Visão Geral e Diagrama de Componentes

A arquitetura da Guardiã AI é estruturada em camadas modulares com separação estrita de responsabilidades:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Interface Gradio (6 Abas Interativas)                     │
│  Tab 1: Chat Livre    Tab 2: Triagem       Tab 3: Violência (LGPD)           │
│  Tab 4: Obstétrico    Tab 5: Prevenção     Tab 6: Risco Gestacional (ML)     │
│  Sidebar: Seleção de Paciente Mock (Faker pt_BR) + Alertas Dinâmicos         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
┌─────────────────────────────────┐        ┌───────────────────────────────────┐
│     Agente ReAct (LangChain)    │        │    5 Workflows LangGraph          │
│  - Tool calling determinístico  │        │  - Triagem Ginecológica           │
│  - Síntese de orientações       │        │  - Detecção de Violência          │
└────────────────┬────────────────┘        │  - Atendimento Obstétrico         │
                 │                         │  - Prevenção e Rastreamento       │
                 │                         │  - Risco Gestacional (ML)         │
                 │                         └─────────────────┬─────────────────┘
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│                      10 Ferramentas Estruturadas (Tools)                     │
│   consultar_prontuario       exames_atrasados          historico_exames      │
│   consultar_medicamento      calendario_menstrual      registrar_violencia   │
│   consultar_violencia (LGPD) buscar_protocolo          avaliar_padrao_viol   │
│   predizer_risco_gestacional (Integração ML)                                 │
└──────┬───────────────────────┬─────────────────────────────┬─────────────────┘
       │                       │                             │
┌──────▼──────┐         ┌──────▼──────────────┐       ┌──────▼─────────────────┐
│  Banco Mock │         │      RAG Híbrido    │       │     Subsistema ML      │
│   SQLite    │         │  - CPU: RAG Lexical │       │  - Regressão Logística │
│ (8 tabelas) │         │    (protocolos json)│       │  - Pipeline Scikit     │
│             │         │  - GPU: Chroma DB   │       │  - Explicabilidade     │
└─────────────┘         └─────────────────────┘       └────────────────────────┘
       ▲                                                             ▲
       │                                                             │
┌──────┴─────────────────────────────────────────────────────────────┴─────────┐
│                             Ambiente & Deploy                                │
│   - Dockerfile: perfil demo-cpu com scripts/app.py e healthcheck             │
│   - docker-compose.yml: container guardiana-ai-app na porta 7860             │
│   - Scripts de Inicialização: start.ps1 (PowerShell) e start.bat (Windows)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes do Sistema

### 2.1 Interface de Usuário (Gradio Blocks)
A UI foi construída com `gr.Blocks` no módulo [`lib/ui.py`](lib/ui.py), organizada em 6 abas especializadas:
- **Sidebar Dinâmica:** Permite alternar pacientes fictícios (50 gerados via Faker pt_BR), exibindo dados cadastrais, resumo ginecológico e alertas imediatos de exames preventivos em atraso.
- **Tab 1 — Consulta Livre:** Chat com agente LangChain ReAct para dúvidas clínicas e consulta a protocolos e prontuários.
- **Tab 2 — Triagem Ginecológica:** Workflow LangGraph que processa sintomas, identifica sinais de gravidade e formula hipóteses diagnósticas com apoio de RAG.
- **Tab 3 — Detecção de Violência Doméstica:** Checklist heurístico de sinais de alerta e workflow de acolhimento seguro, gerando orientações para notificação compulsória (SINAN) e acionamento da rede de proteção (Ligue 180, CAPS, etc.).
- **Tab 4 — Atendimento Obstétrico:** Workflow LangGraph com cálculo de idade gestacional (DUM), identificação de sinais de alarme e integração opcional com classificação de risco.
- **Tab 5 — Prevenção e Rastreamento:** Consulta histórico de exames preventivos e gera plano personalizado de rastreamento conforme diretrizes do Ministério da Saúde.
- **Tab 6 — Risco Gestacional (ML):** Interface para entrada de 11 features clínicas obrigatórias (com opção de carga rápida da paciente selecionada), execução da inferência de Machine Learning, visualização da probabilidade calibrada, explicabilidade de features e acionamento de regras de bypass de emergência.

---

### 2.2 Subsistema de Machine Learning (Estratificação de Risco Gestacional)

Desenvolvido para substituir a classificação heurística/LLM por um classificador tabular auditável e reprodutível:

| Atributo | Descrição |
|---|---|
| **Problema** | Classificação binária: Risco Habitual (0) vs. Alto Risco Gestacional (1) |
| **Dataset** | 8.000 amostras sintéticas com 24 features clínicas (`artifacts/data/risco_gestacional_v1.parquet`) geradas com semente 42 |
| **Integridade** | Manifesto SHA-256 verificado em `artifacts/data/risco_gestacional_v1.manifest.json` |
| **Métrica Principal** | PR-AUC (área sob a curva Precision-Recall), adequada ao desbalanceamento (~20,6% de alto risco) |
| **Modelo Vencedor** | **Regressão Logística** com PR-AUC de **0,590** no teste (superando Random Forest com 0,557 e baseline por regras com 0,263) |
| **Limiar Operacional** | **0,278** (definido na validação para maximizar o recall na classe positiva: recall+ = 0,955) |
| **Explicabilidade** | Pesos lineares no CPU (`coef_linear`) e importância global em `artifacts/explainability/importancia_global.json` |
| **Auditoria** | Cada inferência gera registro na tabela SQLite `predicoes_ml` com hash das features de entrada, sem expor dados sensíveis em texto claro |

---

### 2.3 Orquestração com LangGraph

O sistema implementa 5 StateGraphs explícitos:
1. `triagem`: Coleta dados → identifica gravidade → consulta protocolos RAG → sintetiza conduta.
2. `violencia`: Analisa relatos/sinais → avalia risco iminente → orienta SINAN e rede de proteção → audita acesso em `log_acesso`.
3. `obstetrico`: Coleta dados gestacionais → avalia risco → se `ML_RISCO_HABILITADO=1`, executa nó de ML → verifica sinais de alarme → programa exames.
4. `prevencao`: Verifica exames pendentes (Papanicolau trienal 25-64a, Mamografia bienal 50-69a) → emite alertas e plano de cuidado.
5. `risco_ml`: Workflow completo com 4 caminhos condicionais:
   - **Normal:** Validação de features → inferência ML → explicabilidade → RAG → síntese supervisionada.
   - **Bypass de Regra:** Presença de sinais graves de emergência (ex.: cefaleia intensa + escotomas + PA elevada) encaminha imediatamente para alto risco, sobrepondo o modelo estatístico.
   - **Incompleto:** Falta de campos mandatórios interrompe a inferência e solicita preenchimento.
   - **Degradado:** Em caso de indisponibilidade de artefatos de ML, utiliza baseline determinístico de segurança com aviso explícito.

---

### 2.4 Catálogo de Ferramentas Estruturadas (10 Tools)

| Tool | Assinatura Resumida | Função Principal |
|---|---|---|
| `consultar_prontuario` | `(paciente_id: int)` | Consulta histórico e dados ginecológicos/obstétricos |
| `exames_atrasados` | `(paciente_id: int)` | Identifica atrasos de Papanicolau, Mamografia e USG |
| `historico_exames` | `(paciente_id: int, tipo: Optional[str])` | Retorna resultados e datas de exames anteriores |
| `consultar_medicamento` | `(nome_ou_indicacao: str)` | Consulta indicações e classificação FDA gestação/lactação |
| `calendario_menstrual` | `(paciente_id: int)` | Calcula DUM, extensão dos ciclos e próxima previsão |
| `registrar_violencia` | `(paciente_id, tipo, encaminhamentos)` | Insere relato com log obrigatório e indicação SINAN |
| `consultar_violencia` | `(paciente_id, motivo: str)` | **Auditada via LGPD:** requer justificativa clínica (≥ 5 caracteres) |
| `buscar_protocolo` | `(query: str, categoria: Optional[str])` | Recupera trechos e metadados de normas clínicas via RAG |
| `avaliar_padrao_violencia` | `(relato: str)` | Avalia checklist heurístico de sinais suspeitos |
| `predizer_risco_gestacional` | `(dados_clinicos: dict)` | Executa inferência do modelo tabular de ML e retorna probabilidade e nível |

---

### 2.5 Banco de Dados Relacional (SQLite Mock — 8 Tabelas)

Arquivo: `artifacts/demo/hospital_ui.db` (ou configurável via `HOSPITAL_DB_PATH`):

1. `pacientes`: Cadastro básico (50 pacientes com nomes fictícios Faker pt_BR e hash de CPF).
2. `prontuario_gineco`: DUM, menarca, fórmula gestacional (G/P/A), métodos contraceptivos e antecedentes.
3. `exames`: Histórico de exames preventivos (Papanicolau, Mamografia, USG) com resultados e datas recomendadas.
4. `registros_violencia`: Registro sigiloso de violência doméstica e notificação SINAN.
5. `log_acesso`: **Auditoria LGPD** registrando usuário, timestamp, paciente consultada e motivo clínico da consulta.
6. `medicamentos`: Base de referência farmacológica com categorias de risco gestacional/lactação.
7. `ciclos_menstruais`: Registro de ciclos, datas de início, duração e sintomas relatados.
8. `predicoes_ml`: Registro de auditoria contendo `correlation_id`, hash SHA-256 das features, modelo utilizado, probabilidade e timestamp.

---

### 2.6 Retrieval-Augmented Generation (RAG)

- **Perfil Local / CPU / Docker:** Utiliza o retriever lexical determinístico [`lib/rag_local.py`](lib/rag_local.py) baseado no corpus versionado [`artifacts/rag/protocolos_subset.json`](artifacts/rag/protocolos_subset.json). Garante respostas com citações formais (`doc_id` e categoria) sem depender de GPU ou bibliotecas nativas de vetorização pesadas.
- **Perfil Colab / GPU:** Utiliza Chroma persistido com embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` sobre os 39 PDFs completos indexados (1.392 chunks).

---

### 2.7 Estratégia de Containerização e Execução

O sistema suporta múltiplos perfis de execução conforme a ADR-005:

- **Dockerfile:** Imagem enxuta baseada em `python:3.12-slim`, instalando dependências sem pacotes CUDA. Configurada com `HEALTHCHECK` e comando padrão `CMD ["python", "scripts/app.py"]` para expor a interface web na porta **7860**.
- **Docker Compose:** Serviço `demo` com mapeamento de volume para `./artifacts`, garantindo persistência dos bancos e métricas.
- **Scripts de Automação:**
  - `start.ps1`: Script PowerShell com detecção de Docker e suporte a `-Mode local`.
  - `start.bat`: Script Batch para inicialização por duplo clique ou terminal Windows sem atrito de execução.

```bash
# Execução direta com Docker
docker compose up --build

# Ou via script auxiliar
.\start.bat
```
