from fastapi.testclient import TestClient


from ..app.app import app
print("9lawi ")

client = TestClient(app)


def test_main_page():
    response = client.get("/")
    assert response.status_code == 200


def test_download():
    response = client.get("/download")
    assert response.status_code == 200

def test_acceuil():
    response = client.get("/Acceuil")
    assert response.status_code == 200
