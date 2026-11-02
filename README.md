# API FastAPI + YOLO + OpenCV

API mínima para detectar pessoas e itens de EPI em imagens.

## Instalação

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Modelo

Por padrão, a API usa `best_ppe_yolo11s.pt`, um peso treinado com as classes `helmet`, `gloves`, `vest`, `boots`, `goggles`, `Person` e classes de ausência como `no_helmet`.

```powershell
$env:MODEL_PATH="best_ppe_yolo11s.pt"
$env:CONFIDENCE="0.35"
```

## Executar

```powershell
uvicorn app.main:app --reload
```

A documentação interativa fica em http://127.0.0.1:8000/docs.

## Testar

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect" -F "file=@foto.jpg"
```

Resposta:

```json
{
  "filename": "foto.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.93,
      "box": {"x1": 10, "y1": 20, "x2": 300, "y2": 500}
    }
  ]
}
```
