import sys
import os 



from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import sys
import os
from app import app, decision
from fastapi import status
import pytest



# Client test
client = TestClient(app)

def test_main_page():
    response = client.get("/")
    assert response.status_code == 200

# async_client = AsyncClient()
# @pytest.mark.asyncio
# def test_web_socket():
    """
    In this function, we test the websocket connection used for
    streaming the models' output
    """
    # response = await  async_client.get("/ws")
    # assert response.status_code == status.HTTP_200_OK


    # with client.websocket_connect('/ws') as websocket:
    #     pass






def test_decision():
    """
    In this function, we test decision function  which takes as inputs
    an image and coordinates of the detected box. It returns the
    coordinates of the centred box
    """

    # Input image
    image = Image.open("./savedimage.jpg")

    # THe box is well centred, the position's camera will not change
    list_input_no_change = [173.0, 281.0, 460.0, 480.0]
    coord_centred_box_no_change = [176.50000, 140.50000, 463.50000, 339.50000]
    no_change_decision = " NO CHANGE -- "

    # Check the observed and expected outputs
    coord_centre_box, output = decision(np.array(image), list_input_no_change)
    assert [ele.tolist() for ele in coord_centre_box] == coord_centred_box_no_change
    assert output == no_change_decision

    # The box detected is on the left, the camera should turn right
    list_input_right = [194.0, 280.0, 490.0, 479.0]
    coord_centred_box_right = [172.00000, 140.50000, 468.00000, 339.50000]
    right_decision = " TURN RIGHT -- "

    # Check the observed and expected outputs
    coord_centre_box, output = decision(np.array(image), list_input_right)
    assert [ele.tolist() for ele in coord_centre_box] == coord_centred_box_right
    assert output == right_decision

    # The box detected is on the right, the camera should turn left
    list_input_left = [151.0, 281.0, 391.0, 480.0]
    coord_centred_box_left = [200.00000, 140.50000, 440.00000, 339.50000]
    left_decision = " TURN LEFT -- "

    # Check the observed and expected outputs
    coord_centre_box, output = decision(np.array(image), list_input_left)
    assert [ele.tolist() for ele in coord_centre_box] == coord_centred_box_left
    assert output == left_decision






def test_acceuil():
    response = client.get("/Acceuil")
    assert response.status_code == 200



def test_upload_file():
    """
    In this function, we test "uploader_" route when uploading a file
    """
    # our input image
    image = "./zidane.jpg"

    # Post the input image on the "uploader" route
    response = client.post(
        "/uploader_", files={"file_1": ("filename", open(image, "rb"), "image/jpg")}
    )
    assert response.status_code == 200




def test_download():
    """
    In this function, we download the result after generating it with test_upload_file
    """
    response = client.get("/download")
    assert response.status_code == 200






