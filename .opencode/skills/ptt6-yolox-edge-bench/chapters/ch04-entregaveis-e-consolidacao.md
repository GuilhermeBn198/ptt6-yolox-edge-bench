# Capítulo 4: Entregáveis e Consolidação

## Ideia Central
O projeto se comprova por **cinco entregáveis-base obrigatórios** e dois produtos explícitos (relatório técnico como principal, dataset CSV como secundário), com consolidação pós-disciplina em evento regional e DOI (Zenodo).

## Frameworks Introduzidos
- **Produtos**:
  - **Produto principal**: relatório técnico conclusivo.
  - **Produto secundário**: dataset publicado (medições brutas em CSV).
- **Entregáveis-base (obrigatórios para qualquer trilha)**:
  1. Repositório versionado com licença.
  2. README de reprodutibilidade.
  3. Dados brutos (CSV com o resultado das execuções).
  4. Vídeo de demonstração (3–5 min) das execuções nas duas plataformas (ESP32-S3 e RPi Zero 2 W).
  5. Relatório técnico conclusivo com tabelas e gráficos comparativos.
- **Consolidação pós-disciplina**:
  - Submissão a evento regional de sistemas computacionais.
  - Publicação do dataset de medições com **DOI (Zenodo)**.
- **Referências preliminares**:
  - Patterson & Hennessy — *Computer Organization and Design* (edição da disciplina).
  - Datasheets das plataformas (ESP32; Raspberry Pi Zero 2 W).
  - Repositório YOLOX — https://github.com/Megvii-BaseDetection/YOLOX

## Conceitos-Chave
- **Entregável-base**: item obrigatório, comum a todas as trilhas de PTT.
- **README de reprodutibilidade**: documento que permite a terceiros repetir o experimento.
- **Dados brutos (CSV)**: resultado das execuções, sem pós-processamento que esconda variabilidade.
- **Vídeo de demonstração (3–5 min)**: prova visual das execuções nas duas plataformas.
- **Relatório técnico conclusivo**: documento com tabelas e gráficos comparativos.
- **DOI (Zenodo)**: identificador persistente para citar o dataset publicado.
- **Licença**: termo de uso do repositório (requisito do entregável 1).

## Modelos Mentais
- Use **"reprodutibilidade por terceiros"** como teste final: se alguém não consegue repetir com o README + CSV, o entregável falhou.
- Pense no **vídeo como evidência**, não como marketing: mostre as duas plataformas executando de fato.
- Trate o **CSV bruto** como fonte da verdade: gráficos e estatísticas derivam dele, nunca o substituem.

## Anti-padrões
- **Publicar só dados agregados**: impede reanálise e fere o entregável de dados brutos.
- **Vídeo demonstrando apenas uma plataforma**: não cumpre "execuções nas duas plataformas".
- **Repositório sem licença**: falha o entregável-base 1.
- **README sem passos de reprodução**: falha o entregável-base 2 e a consolidação futura.

## Exemplo Trabalhado
Checklist de entrega final (semana 14–15):
1. Repositório público com **LICENSE** e **README de reprodutibilidade** (instruções nas duas plataformas).
2. `data/raw/*.csv` com as execuções (30 por plataforma, todas as colunas: plataforma, execução, latência, throughput, energia).
3. `analysis/` com scripts que geram tabelas/gráficos a partir do CSV bruto.
4. Vídeo de 3–5 min mostrando execução no ESP32-S3 **e** no RPi Zero 2 W.
5. Relatório técnico (PDF/MD) com tabelas comparativas e gráficos.
6. (Pós-disciplina) depositar o CSV no Zenodo e submeter ao evento regional.

## Principais Lições
1. São **cinco entregáveis-base**; nenhum é opcional.
2. Relatório é o produto **principal**; dataset CSV é o **secundário**.
3. Reprodutibilidade (README + CSV bruto) é o critério de qualidade central.
4. Há continuidade planejada: evento regional + DOI.

## Conexões
- **Ch 2**: os entregáveis derivam do protocolo executado.
- **Ch 3**: riscos mitigados protegem estes entregáveis.
- **Patterson & Hennessy**: base conceitual de arquitetura usada na análise (Lei de Amdahl, hierarquia de memória).
