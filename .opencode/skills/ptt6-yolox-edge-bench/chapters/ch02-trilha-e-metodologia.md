# Capítulo 2: Trilha, Fundamentação e Metodologia

## Ideia Central
A trilha **PTT6 — Benchmarking e eficiência energética** define um experimento **controlado de duas plataformas**, com protocolo declarado, medição de energia e análise estatística comparativa.

## Frameworks Introduzidos
- **Escolha da trilha PTT6**: benchmarking e eficiência energética.
  - Quando usar: é a moldura metodológica de todo o projeto.
  - Como: o produto principal é um **relatório técnico conclusivo**; o secundário é um **dataset bruto em CSV**.
  - Justificativa: Edge AI é a temática dos trabalhos finais dos discentes no mestrado.
- **Escopo mínimo viável**:
  - Quando usar: como linha de corte do que precisa existir para entregar.
  - Como: 1 modelo (YOLOX-Nano int8) × 2 plataformas (ESP32 LX6, RPi Zero 2 W); protocolo declarado; **repetições por piloto/CV (piso 30)**; energia **estimada por software**; dataset de detecção de EPIs.
- **Escopo excluído explicitamente**:
  - Comparação com GPU RTX 3090; múltiplos modelos/datasets; otimização/tuning da quantização; inferência em nuvem; implantação em campo prolongada.
- **Protocolo experimental** (ver cheatsheet para tabela; decisões finais em `docs/protocolo.md` §6/§12):
  - **Independente**: plataforma (ESP32 LX6, RPi Zero 2 W)
  - **Dependentes**: latência (ms), throughput (inf/s), energia por inferência (mJ)
  - **Repetições/warm-up**: **piloto** estima o CV → define N (IC 95%), **piso 30**; warm-up **definido pelo dado**
  - **Dispersão**: mediana + IQR e média ± dp; IC 95%
  - **Instrumentos**: energia **estimada por software** (modelo de potência × tempo por fase); cronometragem por **duas bordas de GPIO** (ou timestamps de alta resolução)
  - **Estatística**: Mann-Whitney U + tamanho de efeito + IC bootstrap
  - **Ameaça à validade**: variação térmica → pausas de resfriamento + ambiente registrado
- **Vínculo com a ementa**:
  - **Desempenho e Lei de Amdahl** — comparação de tempos de execução entre arquiteturas.
  - **Hierarquia de memória e E/S** — restrições de memória da MCU frente ao SBC.
  - **Paralelismo** (secundário) — o SBC tem múltiplos núcleos, **não explorados no escopo mínimo**.

## Conceitos-Chave
- **Protocolo declarado**: registro prévio das decisões experimentais para garantir reprodutibilidade.
- **Pipeline funcionalmente idêntico**: mesma tarefa/entrada/saída lógica nas duas plataformas, mesmo que a implementação difira.
- **Aquecimento (warm-up)**: primeiras execuções descartadas por instabilidade inicial (cache, clock, temperatura).
- **Bateria de execução**: bloco de repetições; pausas de resfriamento entre baterias.
- **Coeficiente de variação (CV)**: razão entre desvio-padrão e média — medida de estabilidade relativa.
- **IQR**: intervalo interquartil, robusto a outliers.
- **Energia por software**: `E = Σ (P_modo × t_fase)`, com potência típica de datasheet e tempo medido por fase — substitui o medidor físico inline.
- **Toggle de GPIO**: sinal digital acionado no início/fim da inferência para cronometragem precisa.
- **Inferências por joule**: métrica normalizada de eficiência energética (o "resultado" comparável).

## Modelos Mentais
- Use o **escopo mínimo** como contrato: implemente exatamente o que ele exige antes de qualquer extra.
- Pense no **protocolo** como código congelado: mude uma variável (ex.: nº de repetições) e você invalida a comparação.
- Pense na **ameaça térmica** como ruído sistemático: se o dispositivo esquenta ao longo da sessão, a latência e a energia derrapam juntas.

## Anti-padrões
- **Mudar o protocolo no meio da coleta**: quebra a comparabilidade e invalida H1.
- **Ignorar o warm-up**: medir com o dispositivo ainda instável (cache/clock/temperatura) introduz viés.
- **Medir energia só no SBC ou só na MCU**: sem os dois lados, não há comparação.
- **Confundir throughput com eficiência**: mais rápido ≠ mais eficiente por joule.

## Exemplo Trabalhado
Decisão: **quantas repetições e como calcular a métrica final?**
1. Rodar um **piloto** e estimar o **CV** de latência/energia por plataforma.
2. Calcular **N** para IC de 95% (piso 30); o warm-up é **definido pelo dado** (variação < 2%, piso 5).
3. Calcular **mediana + IQR** e **média ± dp** de latência, throughput e energia.
4. Converter energia por inferência em **inferências/joule** para comparação normalizada.
5. Comparar ESP32 LX6 vs RPi Zero 2 W (Mann-Whitney U) e testar H1 (throughput do SBC ↑; energia do SBC ↑/igual).
Registrar o passo a passo no README para reprodutibilidade.

## Principais Lições
1. A trilha PTT6 define produto principal (**relatório**) e secundário (**dataset CSV**).
2. O escopo mínimo é o contrato de entrega; o escopo excluído evita deriva.
3. O rigor está no protocolo: piloto→N, warm-up por dado, mediana/IQR, energia por software, GPIO; decisões finais em `docs/protocolo.md`.
4. A comparação só vale se o pipeline for **funcionalmente idêntico**.

## Conexões
- **Ch 1**: operacionaliza a pergunta de pesquisa e H1.
- **Ch 3**: define hardware, cronograma e riscos que viabilizam o protocolo.
- **Ch 4**: protocolo executado → tabelas/gráficos do relatório e CSV publicado.
