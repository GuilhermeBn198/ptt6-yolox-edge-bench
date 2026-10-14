# Cheatsheet — PTT6 YOLOX Edge Benchmark

## Regras de decisão
- **Nova tarefa?** Só entra se produzir latência/throughput/energia nas duas plataformas, senão → fora de escopo.
- **Ferramenta só existe numa plataforma?** Encontre equivalente funcional ou cai no Plano B; não compare maçãs com laranjas.
- **Energia por software não bate com a realidade?** Não troque o método: registre as premissas (`P_modo`, tensão); se houver medidor, faça **verificação cruzada** e **declare a limitação**.
- **Modelo não roda em algum device?** Troque por outra versão quantizada do YOLO, mantendo o protocolo.
- **Vai para o tempo real antes de validar?** Não. A câmera é **fase 2**; valide com **imagens estáticas** primeiro.
- **Vai mudar o protocolo no meio da coleta?** Não. Congele-o; mudança invalida H1.
- **Resultado surpreendente?** Verifique primeiro a ameaça térmica e o aquecimento, depois a hipótese.

## Protocolo em uma linha
1 modelo (YOLOX-Nano int8) × 2 plataformas (ESP32 LX6, RPi Zero 2 W) → piloto→N (piso 30) → warm-up por dado → mediana, IQR, IC → inferências/joule.

## Variáveis
| Tipo | Variável | Unidade |
|------|----------|---------|
| Independente | Plataforma (ESP32 (LX6) / RPi Zero 2 W) | — |
| Dependente | Latência | ms |
| Dependente | Throughput | inferências/s |
| Dependente | Energia por inferência | mJ |

## Hipótese H1
> SBC throughput **>** MCU; SBC energia/inferência **≥** MCU (consumo estático maior).

Resultado que **confirma**: throughput(SBC) > throughput(MCU) **e** energia(SBC) ≥ energia(MCU).
Resultado que **refuta**: throughput(SBC) ≤ throughput(MCU) **ou** energia(SBC) < energia(MCU).

## Cronograma (15 semanas)
| Semanas | Fase | Bloqueia |
|---------|------|----------|
| 1–2 | Revisão + protocolo | implementação |
| 3–5 | Implementação do pipeline | experimentos |
| 6–9 | Coleta (N por piloto; piso 30) | análise |
| 10–11 | Análise estatística | redação |
| 12–13 | Relatório técnico | vídeo/entrega |
| 14–15 | Vídeo + ajustes finais | entrega |

## Limiares & padrões
- Repetições: **piloto→N** (IC 95%), **piso 30**; warm-up **definido pelo dado** (piso 5).
- Energia: **estimada por software** (`E = Σ P_modo · t_fase`); cronometragem por **GPIO** (2 bordas).
- Fases: **F1 estático → F2 câmera → F3 medição → F4 análise**.
- Vídeo: **3–5 min** (mostrar as duas plataformas).
- Medida de energia: **joule**; eficiência: **inferências/joule**.

## Riscos → resposta
| Risco | Prob. | Impacto | Resposta |
|-------|-------|---------|---------|
| Energia por software não refletir o consumo real | média | alto | mesmo modelo nas duas pontas; registrar premissas; verificação cruzada; declarar limitação |
| Modelo não performa em um device | média | alto | Outras versões quantizadas do YOLO |
| Câmera não integrar na fase 2 | baixa | médio | manter validação estática como evidência; declarar a fase não concluída |

## Tells & smells
- Latência subindo ao longo da sessão → **aquecimento térmico**; insira pausas.
- Energia por inferência quase igual entre plataformas → suspeite do consumo **estático** do SBC dominando.
- Números "limpos demais" → verifique se os dados brutos foram preservados (não só médias).
- Alguém propondo 2º modelo/GPU/nuvem → **deriva de escopo**; registre como trabalho futuro.

## Checklist de entrega
- [ ] Repositório com licença
- [ ] README de reprodutibilidade
- [ ] CSV bruto das execuções
- [ ] Vídeo (3–5 min) nas duas plataformas
- [ ] Relatório técnico com tabelas/gráficos
- [ ] (pós) dataset no Zenodo com DOI + submissão a evento regional
