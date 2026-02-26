# Documento de Cálculos e Requisitos de Hardware: Inspeção < 800ms

## 1. Introdução e Definição do Problema
O objetivo é que o tempo total entre o trigger de uma câmera **Line Scan de 8K** e a resposta final da inspeção de **48 latas** seja estritamente inferior a **800 milissegundos (ms)**.

Este documento detalha o fracionamento do tempo, as necessidades computacionais de cada etapa e projeta a arquitetura de hardware necessária para suportar essa cadência em ambiente industrial.

---

## 2. Análise do Fluxo de Tempo (Time Budget)
Temos exatos 800 ms para processar toda a rotina. A resolução 8K garante imagens muito grandes — vamos assumir que a matriz completa (48 latas, mais o fundo) gere uma imagem de aproximadamente 8192 × 8192 pixels (cerca de 67 Megapixels).

### 2.1 Fracionamento Teórico do Pipeline de 800ms:
* **Aquisição e Transferência DMA (Trigger ao Frame na RAM):** ~50 ms
* **Pré-Processamento Global (Detecção de Cantos, Retificação/Warp):** ~100 ms
* **Recorte das 48 Latas (Crop):** ~20 ms
* **Cálculo de Overhead e Reserva de Segurança:** ~100 ms
* **Tempo Restante para Processamento Individual (48 latas):** ~530 ms

### 2.2 O Desafio dos 530 ms
Com 530 ms restantes na margem de segurança para lidar com o processo de "Alinhamento (SIFT/ECC) + Processamento (CLAHE/Resize) + Inferência de IA (OpenVINO)", temos a seguinte janela por lata (caso feito sequencialmente):
* **530 ms / 48 latas ≈ 11.0 ms por lata.**

Fazer crop, alinhamento robusto sub-pixel, equalização de histograma e a inferência de um modelo de Inteligência Artificial em **11 ms por unidade** em um laço sequencial é na prática impossível usando operações padrão via CPU e Python puro, principalmente devido ao limite da Thread e o GIL. 

**Solução:** Processamento Paralelo (Multithreading para alinhamento e Processamento em *Batch* para a Inferência IA).

---

## 3. Especificando o Hardware Necessário

Dado o orçamento estrito de tempo acima, hardware potente de processamento paralelo é mandatório. Abaixo, a sugestão da máquina visionária.

### 3.1. Processador (CPU)
O gargalo na retificação de uma imagem de 67 MP e no processamento SIFT/ECC simultâneo reside na largura de banda da memória (LBM) e núcleos físicos.
* **Recomendação Mínima:** Intel Core i7 / i9 (13ª ou 14ª Geração, e.g., i9-14900K) ou AMD Ryzen 9 7900X / 7950X.
* **Por quê?** Mínimo de 16 a 24 núcleos. Precisamos rodar o alinhamento de forma puramente multitarefa (Ex: *ThreadPoolExecutor* no Python alocando as 48 latas simultaneamente pelos threads). As altas frequências (> 5.0 GHz) da CPU garantem que o tempo de overhead de troca de contexto da linguagem não consuma ms cruciais.

### 3.2. Placa Gráfica (GPU) / Acelerador
Se o peso do software está no modelo (EfficientAD / PatchCore), realizar as 48 inferências sequenciais vai arrebentar o orçamento de tempo. O modelo obrigatoriamente fará *"Batch Inference"* — em vez de fazer 48 requisições à IA, enviamos 1 bloco contendo as 48 latas.
* **Recomendação Mínima:** Placa NVIDIA dedicada. **RTX 4070 Ti, RTX 4080**, ou para contextos industriais operando 24/7 sem falha, as **NVIDIA RTX 4000 / 5000 Ada Generation**.
* **Por quê?** Somente aceleradores pesados garantem que um tensor de tamanho [48, Canais, Altura_Lata, Largura_Lata] seja processado por completo por algoritmos de detecção de anomalias modernos em 100-200ms na GPU através de bibliotecas como **TensorRT** do que com o **OpenVINO/CPU**.

### 3.3. Memória RAM (Atenção Máxima)
Uma imagem line scan descompactada com ~67 MP pesa perto de **200 Megabytes**. Manipular matrizes NumPy deste volume consome Gb/s imensos de transferências internas.
* **Recomendação Mínima:** 32 GB a 64 GB de **DDR5**.
* **Velocidade:** Acima de 6000 MT/s, montado estritamente em **Dual-Channel** completo, priorizando baixas latências (CAS Delay agressivo) para garantir que as operações de cópia de buffers (como o Resize e Crop) sejam quase instantâneos. 

### 3.4. Interface e Acesso de Câmera 
O uso de uma Line Scan de 8K exige enorme taxa de dados na entrada pela comunicação.
* **Câmeras GigE Vision:** Obrigatório montar uma Placa PCIe Controladora de Rede dedicada **10 GbE** (10 Gigabit Ethernet). Jamais depender do conector da própria placa mãe.
* **Câmeras CameraLink / CoaXPress:** Utilizar o Frame Grabber proprietário (ex: Matrox, Euresys) exigido nativamente sempre instalado em trilhas de placa-mãe PCIe x8 ou superior, utilizando transferência por Acesso Direto à Memória (DMA), enviando frames da placa sem envolver o ciclo da CPU.

---

## 4. Requisitos de Otimização no Software

Mesmo o melhor hardware falhará o objetivo de 800ms sem adaptações arquiteturais no código Python em ambiente industrial:
1. **Inferência em Lote (Batch Inference):** Ao invés de um laço `for lata in latas: infer()`, o código deve preparar uma lista, formatar numa matriz numpy única `[48, C, H, W]` e chamar `.infer(lote)` de uma vez.
2. **Multiprocessing/Multithreading nas Rotinas OpenCV:** Delegar o `detect_corners()`, `warpPerspective` e `crop` para executores paralelos. Para Alinhamento e Equalização CLAHE (que são intensivas por CPU por matriz), dividir em vários *workers* correspondentes ao tamanho do processador.
3. **Conversões Eficientes:** Eliminar loops em cálculos matemáticos (garantir conversão vetorial estrita do NumPy) e evitar passagens e cópias de variáveis pelo Python como `.copy()` excessivos, criando pré-alocações de memória.

## 5. Resumo da Setup 
* **Motherboard:** Chipset Industrial high-end com slots PCIe Gen 4 / Gen 5 desbloqueados (x16).
* **Processador:** Mínimo de Core i7 / Ryzen 9 (16+ Núcleos de alta frequência).
* **RAM:** 64 GB DDR5 (6000+ MHz operando Dual-Channel).
* **Placa de Vídeo:** NVIDIA RTX 4070 Ti / 4080 (Integrando TensorRT ao invés de usar inferência via CPU OpenVINO pura se falhar nos testes de stress).
* **Armazenamento:** M2 NVMe SSD rápido apenas para o log (embora Logging de imagens reprovadas deverá ser assíncrono para fora da thread de inspeção, sem bloqueios de tempo (ms) perante gravação física no disco).
