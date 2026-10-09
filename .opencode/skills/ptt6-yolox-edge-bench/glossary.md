# Glossário — PTT6 YOLOX Edge Benchmark

**Amdahl (Lei de)** — ganho de desempenho limitado pela fração serial; usada para comparar tempos entre arquiteturas (Ch 2).

**AQ (aquecimento / warm-up)** — primeiras execuções descartadas antes da medição válida (5 primeiras, Ch 2).

**Bateria de execução** — bloco de repetições; pausas de resfriamento ocorrem entre baterias (Ch 2, Ch 3).

**Coeficiente de variação (CV)** — desvio-padrão dividido pela média; mede estabilidade relativa (Ch 2).

**Cronometragem por GPIO** — toggle de um pino no início/fim da inferência para medir tempo (Ch 2).

**Dataset de EPIs** — conjunto de dados para classificação de Equipamentos de Proteção Individual (Ch 1, Ch 3).

**Dispersão (medidas de)** — desvio-padrão, IQR e coeficiente de variação (Ch 2).

**DOI (Zenodo)** — identificador persistente para publicar/citar o dataset (Ch 4).

**Edge AI** — inferência de redes neurais em dispositivos de borda, de baixo custo e consumo restrito (Ch 1).

**Eficiência energética normalizada** — inferências por joule; principal métrica comparável (Ch 1, Ch 2).

**Energia por inferência (mJ)** — energia total dividida pelo número de inferências (Ch 1, Ch 2).

**Entregável-base** — item obrigatório comum a todas as trilhas de PTT (Ch 4).

**ESP32-S3** — MCU alvo, representante do extremo de baixo consumo (Ch 1, Ch 3).

**Escopo mínimo viável** — 1 modelo × 2 plataformas, 30 repetições, energia inline, dataset de EPIs (Ch 2).

**Fonte de alimentação regulada** — garante tensão estável durante a medição (Ch 3).

**H1** — hipótese: SBC tem maior throughput, mas energia por inferência igual ou superior (Ch 1).

**INA219** — medidor de potência inline, amostra a 10 Hz (Ch 2, Ch 3).

**int8** — quantização em inteiro de 8 bits (Ch 1).

**IQR** — intervalo interquartil; medida de dispersão robusta a outliers (Ch 2).

**Latência de inferência (ms)** — tempo de uma inferência (variável dependente) (Ch 1, Ch 2).

**Pipeline funcionalmente idêntico** — mesma lógica de inferência nas duas plataformas (Ch 2).

**Plano B** — caminho alternativo pré-aprovado para riscos mapeados (Ch 3).

**Produto principal** — relatório técnico conclusivo (Ch 4).

**Produto secundário** — dataset publicado em CSV (Ch 4).

**Protocolo experimental** — independentes, dependentes, repetições, dispersão, instrumentos, ameaça (Ch 2).

**Proxy documentado** — medição substituta (ex.: datasheet) com limitação declarada (Ch 3).

**PTT6** — trilha de Benchmarking e eficiência energética (Ch 2).

**Raspberry Pi Zero 2 W** — SBC alvo, representante do extremo de maior throughput/recursos (Ch 1, Ch 3).

**README de reprodutibilidade** — passos que permitem repetir o experimento (Ch 4).

**Throughput (inferências/s)** — inferências por segundo (variável dependente) (Ch 1, Ch 2).

**Toggle de GPIO** — ver cronometragem por GPIO (Ch 2).

**Vídeo de demonstração** — 3–5 min mostrando execução nas duas plataformas (Ch 4).

**YOLOX-Nano** — modelo leve usado nas duas plataformas, quantizado int8 (Ch 1, Ch 2).
