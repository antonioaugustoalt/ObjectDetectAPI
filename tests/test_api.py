
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ObjectDetect API funcionando"
    }


def test_invalid_file():
    response = client.post(
        "/detect",
        files={
            "file": (
                "arquivo.txt",
                b"conteudo de teste",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Formato inválido. Envie JPG ou PNG."



from PIL import Image
import io


def test_detect_valid_image():
    image = Image.new("RGB", (640, 480), color="white")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post(
        "/detect",
        files={
            "file": (
                "teste.jpg",
                image_bytes,
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200
    assert "total_detections" in response.json()
    assert "detections" in response.json()
