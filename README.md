# ObjectDetect API

API de detecção de objetos em imagens utilizando **YOLO11n**, **FastAPI** e o dataset **COCO 2017**.

## 1. Objetivo

Desenvolver uma API REST capaz de identificar objetos em imagens, retornando:

- Nome da classe detectada;
- Nível de confiança da detecção;
- Coordenadas da bounding box.

A API recebe imagens JPG ou PNG e retorna os resultados em formato JSON.

## 2. Tecnologias utilizadas

- Python 3.14
- Ultralytics YOLO 11n
- FastAPI
- Uvicorn
- Pillow
- Pytest

## 3. Dataset

Foi utilizado o **COCO 2017**, especificamente o conjunto de validação `val2017`, composto por 5.000 imagens.

O dataset possui 80 categorias. Para este projeto, foram selecionadas 15 classes:

1. person
2. bicycle
3. car
4. motorcycle
5. bus
6. truck
7. dog
8. cat
9. bird
10. bottle
11. cup
12. chair
13. laptop
14. book
15. backpack

O modelo YOLO11n utiliza pesos pré-treinados no COCO. A API filtra as detecções para retornar somente as 15 classes selecionadas.

O dataset e os pesos do modelo não estão incluídos no repositório.

## 4. Estrutura do projeto

```text
ObjectDetectAPI/
├── app/
│   ├── __init__.py
│   └── main.py
├── dataset/
├── models/
├── tests/
│   └── test_api.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Instalação

Clone o repositório:

```bash
git clone https://github.com/antonioaugustoalt/ObjectDetectAPI.git
cd ObjectDetectAPI
```

Crie e ative o ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 6. Execução

Inicie a API:

```bash
uvicorn app.main:app --reload
```

Acesse a documentação interativa do Swagger:

http://127.0.0.1:8000/docs

Endpoint principal:

`POST /detect`

O endpoint recebe uma imagem por meio de `multipart/form-data`, utilizando o campo `file`.

Formatos aceitos: JPG e PNG.

## 7. Exemplo de resposta JSON

```json
{
  "filename": "imagem.jpg",
  "total_detections": 1,
  "detections": [
    {
      "class_name": "dog",
      "confidence": 0.9117,
      "box": {
        "x1": 120.5,
        "y1": 80.2,
        "x2": 450.7,
        "y2": 390.4
      }
    }
  ]
}
```

Os valores acima são ilustrativos. As coordenadas e a confiança variam conforme a imagem analisada.

## 8. Testes automatizados

Os testes foram implementados utilizando Pytest e FastAPI TestClient.

Para executar:

```bash
python -m pytest -v
```

Testes realizados:

- Verificação do endpoint inicial;
- Rejeição de arquivos em formatos inválidos;
- Processamento de uma imagem válida.

**Resultado:** 3 testes aprovados.

## 9. Limitações

O projeto utiliza um modelo pré-treinado, sem fine-tuning específico para as imagens de validação.

Como consequência, podem ocorrer falsos positivos e classificações incorretas. A confiança apresentada representa a estimativa do modelo, não uma garantia de precisão.

## 10. Kanban

Quadro de acompanhamento do desenvolvimento:


