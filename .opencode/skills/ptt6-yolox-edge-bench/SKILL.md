---
name: ptt6-yolox-edge-bench
description: "Base de conhecimento da proposta do projeto PTT6 \"Benchmarking de inferência do Yolo quantizado em MCU e SBC\" (Filipe Matias & Guilherme Lucas, PPCA/UFRR, orient. Herbert Oliveira Rocha). Use ao planejar, executar, medir ou redigir o benchmark de YOLOX-Nano int8 comparando ESP32-S3 (MCU) e Raspberry Pi Zero 2 W (SBC) para classificação de EPIs — pergunta de pesquisa, hipótese, protocolo experimental, medição de energia, riscos/Plano B e entregáveis."
---

<!-- argument-hint: [tópico, framework ou número de seção] -->

# Benchmarking de Inferência do YOLO Quantizado em MCU e SBC
**Autoria**: Filipe Matias & Guilherme Lucas | **Orientador**: Herbert Oliveira Rocha | **Curso**: Mestrado Profissional em Computação Aplicada (PPCA/UFRR) | **Trilha**: PTT6 | **Prazo**: 11 semanas (cronograma de 15) | **Gerado a partir de**: PTT6_Filipe_Guilherme_assinado_assinado.pdf

> Documento-fonte: proposta assinada do Projeto de Pesquisa (PTT6 — Benchmarking e eficiência energética).
> Escopo: Edge AI embarcado. Este é um **projeto de medição comparativa**, não de treinamento de modelo.

## How to Use This Skill

- **Sem argumentos** — carregue os frameworks centrais (pergunta, hipótese, protocolo, restrições, entregáveis)
- **Com um tópico** — peça `energia`, `INA219`, `int8`, `dataset EPI`; eu localizo e leio a seção relevante
- **Com seção** — peça `ch02`; carrego o arquivo específico
- **Browse** — pergunte "quais capítulos você tem?" para ver o índice completo

Quando o tópico não estiver nos Frameworks Centrais abaixo, leio o arquivo da seção correspondente antes de responder.

---

## Frameworks Centrais & Modelos Mentais

### Pergunta de pesquisa (o "norte" de toda decisão)
**Como se comparam latência, throughput e energia por inferência** entre uma **MCU (ESP32-S3)** e um **SBC (Raspberry Pi Zero 2 W)** ao executar o **mesmo YOLOX-Nano quantizado int8** para **classificação de EPIs**?
- Use isto como filtro: qualquer tarefa que não sirva a essa comparação está fora de escopo.

### Hipótese H1 (o que você está tentando provar/refutar)
**H1**: o SBC apresenta **throughput maior** que a MCU, mas com **energia por inferência igual ou superior**, dado seu **consumo estático mais alto**.
- Preferir comparar **inferências por joule** (eficiência normalizada) e não só velocidade — a MCU pode "ganhar" em eficiência mesmo perdendo em throughput.

### Escopo mínimo viável (a linha que não se cruza)
Um único modelo (YOLOX-Nano int8) × exatamente duas plataformas (ESP32-S3 e RPi Zero 2 W), com **protocolo declarado**, **30 repetições por plataforma** e **medição de energia por medidor inline** (ou proxy documentado). Dataset de classificação de EPIs.

### Escopo excluído (diga "não" cedo)
GPU RTX 3090 · múltiplos modelos/datasets · tuning/otimização da quantização · inferência em nuvem · implantação em campo prolongada.

### Protocolo experimental (o núcleo do rigor)
- **Independente**: plataforma (ESP32-S3, RPi Zero 2 W)
- **Dependentes**: latência (ms), throughput (inf/s), energia por inferência (mJ)
- **Repetições**: 30 por plataforma, **descartando as 5 primeiras** (aquecimento)
- **Dispersão**: desvio-padrão, IQR e coeficiente de variação (CV)
- **Instrumentos**: medidor de potência inline (ex.: INA219) amostrando a **10 Hz**; cronometragem por **toggle de GPIO**
- **Ameaça principal**: variação térmica → mitigar com pausas de resfriamento entre baterias e ambiente refrigerado

### Regra da comparabilidade
Ferramentas e pipeline **funcionalmente idênticos** nas duas plataformas. Se o mesmo artefato (modelo/pipeline) não roda igual nos dois lados, a comparação é inválida — cai no Plano B.

### Regra do Plano B (nunca bloqueie a entrega)
- **Medidor físico indisponível/defeituoso** (prob. média, impacto alto): **estimar energia a partir do datasheet** × tempo de execução medido, **declarando a limitação** no relatório.
- **Modelo não performa em algum dispositivo** (prob. média, impacto alto): usar **outras versões quantizadas do YOLO**.

### Entregáveis-base (obrigatórios)
Repositório versionado com licença · README de reprodutibilidade · **dados brutos em CSV** · **vídeo de demonstração (3–5 min)** nas duas plataformas · **relatório técnico conclusivo** com tabelas e gráficos.
- **Produto principal**: relatório técnico. **Produto secundário**: dataset publicado (CSV).
- **Consolidação pós-disciplina**: submissão a evento regional + dataset com DOI (Zenodo).

### Vínculo com a ementa (por que a disciplina importa)
Desempenho e **Lei de Amdahl** (comparar tempos entre arquiteturas) · **hierarquia de memória e E/S** (restrição de RAM da MCU vs SBC) · secundariamente **paralelismo** (núcleos do SBC, não explorados no escopo mínimo).

---

## Chapter Index

| # | Título | Frameworks-chave |
|---|--------|------------------|
| [ch01](chapters/ch01-contexto-e-problema.md) | Contexto, Problema e Objetivos | Pergunta de pesquisa, H1, objetivos geral/específicos, motivação Edge AI |
| [ch02](chapters/ch02-trilha-e-metodologia.md) | Trilha, Fundamentação e Metodologia | Escolha PTT6, vínculo com ementa, escopo mínimo/excluído, protocolo experimental |
| [ch03](chapters/ch03-infraestrutura-e-execucao.md) | Infraestrutura, Cronograma e Riscos | Hardware, kit, cronograma de 15 semanas, riscos e Plano B |
| [ch04](chapters/ch04-entregaveis-e-consolidacao.md) | Entregáveis e Consolidação | Entregáveis-base, vídeo, relatório, dataset/DOI, referências |

## Topic Index

- **Amdahl / desempenho** → ch02
- **Dataset de EPIs** → ch01, ch03
- **Energia por inferência (mJ)** → ch02, ch03
- **Entregáveis / vídeo / relatório** → ch04
- **ESP32-S3 (MCU)** → ch01, ch02, ch03
- **Escopo excluído** → ch02
- **H1 / hipótese** → ch01
- **INA219 / medidor inline** → ch03
- **int8 / quantização** → ch01, ch02
- **Latência / throughput** → ch01, ch02
- **Lei de Amdahl** → ch02
- **Plano B / riscos** → ch03
- **Raspberry Pi Zero 2 W (SBC)** → ch01, ch02, ch03
- **YOLOX-Nano** → ch01, ch02

## Supporting Files

- [glossary.md](glossary.md) — termos-chave com definições
- [patterns.md](patterns.md) — técnicas e padrões do protocolo experimental
- [cheatsheet.md](cheatsheet.md) — regras de decisão, cronograma e limiares

---

## Scope & Limits

Esta skill cobre apenas o conteúdo da **proposta** PTT6 (documento assinado). Ela descreve o plano — não há resultados medidos ainda. Para implementação, combine com as ferramentas do repositório (`firmware/`, `raspberry/`, `analysis/`). Para tópicos além da proposta, consulte o orientador ou a literatura.
