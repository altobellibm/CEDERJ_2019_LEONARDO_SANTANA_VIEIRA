# CEDERJ 2019 - LEONARDO SANTANA VIEIRA - ORIENTADOR ALTOBELLI DE BRITO MANTUAN

IMPLEMENTAÇÃO PARALELA EM CUDA PARA O ALGORITMO DUAL SCALING EM DADOS DE ORDEM DE CLASSIFICAÇÃO

Este repositório contém uma implementação paralela do algoritmo Dual Scaling para dados de ordem de classificação para a plataforma CUDA

<br />

## Como criar uma base de dados de ordem de classificação

O repositório contém também uma implementação que permite a criação de uma base de dados de ordem de classificação

## Como compilar (console)

Na pasta do projeto:
```console
cd synth_db_generator
g++ main.cpp -std=c++11 -o main
```
<br />

## Como executar a aplicação (console)

Na pasta do projeto:
```console
cd synth_db_generator
./main
```

Serão solicitados os dados necessários: número de colunas, número de linhas e o nome do arquivo.

Ou:

Na pasta do projeto:
```console
cd synth_db_generator
./main 500 1000 db.dat
```

Para criar uma base de dados com 500 colunas e 1000 linhas chamada db.dat

<br />

## Implementação paralela

## Como compilar (console)

Instale o driver proprietário da Nvidia e o CUDA, além de informar a localização da biblioteca Cusp para a chamada do compilador, no exemplo a pasta da biblioteca está localizada no diretório gpu_dominance_data.

Na pasta do projeto:
```console
cd gpu_dominance_data
nvcc main.cu -lcusolver -lcublas -std=c++11 -o main -I .
```

## Como executar a aplicação (console)

Na pasta do projeto:
```console
cd gpu_dominance_data
./main db.dat
```

Para executar utilizando a base de exemplo criada anteriormente.