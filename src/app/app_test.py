############################################################################################################################
#                                  Author: Anass MAJJI                                                                     #
#                               File Name: app/app.py                                                                      #
#                           Creation Date: May 05, 2022                                                              #
#                         Source Language: Python                                                                          #
#         Repository:           #
#                              --- Code Description ---                                                                    #
#                         FastAPI code for the model deployment                                                            #
############################################################################################################################


############################################################################################################################
#                                                   Packages                                                                 #
############################################################################################################################

import sys
import os

parent_path = os.path.normpath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(os.path.abspath(".."))

import uvicorn
import numpy as np
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi import status, UploadFile, File, FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from shutil import rmtree
import sys
from pathlib import Path
from yolov5.utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.models.common import DetectMultiBackend

import logging

##########################################################################################################################
#                                                 Main Code                                                              #
##########################################################################################################################


# Initialize Fastapi app
app = FastAPI()

script_dir = os.path.dirname(__file__)

# absolute static repo path
st_abs_file_path = os.path.join(script_dir, "static/")

# absolute html repo path
html_abs_file_path = os.path.join(script_dir, "templates/")


# define templates and static folders
templates = Jinja2Templates(directory=html_abs_file_path)
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")


# load yolov5 model
model = torch.hub.load("yolov5", "custom", path="yolov5s.pt", source="local")

# detect only persons
model.classes = 0

# model is finished to load
print("model loeaded ... ")


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
"""
ROOT = os.path.dirname(os.path.realpath('__file__'))
if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
"""


def decision(image_imread, box_coord):
    """This function computes  the bounding box's position compared to the center of the
    image based on its coordinates. It takes two parameters :
        - image_input : our input image
        - box_coord : list of coordinates of the bounding box
        returns :
            - coord_centre_box : coordinates of the centred bounding box
            - output decision :
                - turn right : if the person detected is on the right of the camera
                - turn left : if the person detected is on the left of the camera
                - No change : the person detected is on the screen center
    """

    # coordinates of the bounding box
    x_min, y_min, x_max, y_max = box_coord[0], box_coord[1], box_coord[2], box_coord[3]

    # image's dimensions
    dim_1 = image_imread.shape[0]
    dim_2 = image_imread.shape[1]

    # Distance between 0 and x_min
    delta_x1 = x_min

    # Distance between dim2 and x_max
    delta_x2 = dim_2 - x_max

    # Distance between 0 and y_min
    delta_y1 = y_min

    # Distance between dim1 and y_max
    delta_y2 = dim_1 - y_max

    # mean of delta 1 and delta 2 following x-axis
    mean_x = (delta_x1 + delta_x2) / 2

    # mean of delta 1 and delta 2 following y-axis
    mean_y = (delta_y1 + delta_y2) / 2

    # Coordinates of the bounding box on the centre of the image
    # x_min and x_max of the bouding box
    x_min_supp = mean_x
    x_max_supp = dim_2 - mean_x

    # y_min and y_max of the bounding box
    y_min_supp = mean_y
    y_max_supp = dim_1 - mean_y

    # coordinates of the bounding box
    coord_centre_box = torch.tensor([x_min_supp, y_min_supp, x_max_supp, y_max_supp])

    # turn_right is equal to True when the person detected is on the right of the camera and is equal to False
    # otherwise
    turn_right = False
    turn_left = False

    # Output of the function : turn right / left or no change
    output = ""

    # If the person detected is on the right of the camera
    if x_min_supp + 15 <= x_min:
        turn_right = True
        turn_left = False
        output = " TURN RIGHT -- "

    # If the person detected is on the left of the camera
    elif x_min <= x_min_supp - 15:
        turn_left = True
        turn_right = False
        output = " TURN LEFT -- "

    # The person detected is on the center of the camera
    elif x_min_supp - 15 < x_min < x_min_supp + 15:
        turn_left = False
        turn_right = False
        output = " NO CHANGE -- "

    return coord_centre_box, output


@app.get("/")
def window_princip(request: Request):
    return templates.TemplateResponse("first_page.html", context={"request": request})


ROOT = ""


@smart_inference_mode()
@app.websocket("/ws")
async def run(
    websocket: WebSocket,
    weights="yolov5s.pt",  # model path or triton URL  ROOT /
    source=0,  # file/dir/URL/glob/screen/0(webcam)
    data="data/coco128.yaml",  # dataset.yaml path  ROOT /
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=0,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="runs/detect",  # save results to project/name  ROOT /
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    await websocket.accept()
    source = str(source)
    print(" source ", source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = (
        source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    )
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    device = select_device(device)
    print("-- devise -- ", device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
        )
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
        )
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    try:
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = (
                    increment_path(save_dir / Path(path).stem, mkdir=True)
                    if visualize
                    else False
                )
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(
                    pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
                )

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}"
                )  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(
                    im0, line_width=line_thickness, example=str(names)
                )

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).tolist()[0]

                        if save_txt:  # Write to file
                            xywh = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                .view(-1)
                                .tolist()
                            )  # normalized xywh
                            line = (
                                (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            )  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = (
                                None
                                if hide_labels
                                else (
                                    names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                                )
                            )
                            # Get the position of the person detected compared to the center of the screen
                            coord_centre_box, finale_decision = decision(
                                im0, [ele.tolist() for ele in xyxy]
                            )

                            annotator.box_label(xyxy, label, color=colors(c, True))

                            # annotate center box
                            # annotator.box_label(coord_centre_box)

                            # print the finale decision : turn right/ left or no change
                            # print("  Finale decision  --- : ", finale_decision)
                            logging.info(finale_decision)

                        if save_crop:
                            save_one_box(
                                xyxy,
                                imc,
                                file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                                BGR=True,
                            )

                # Stream results
                im0 = annotator.result()

                # encode the annotated image
                _, encoded_img = cv2.imencode(".png", im0)
                # encoded_image = base64.b64encode(encoded_img).decode("utf-8")

                # send the output in the websocket to the client
                await websocket.send_bytes(encoded_img.tobytes())

    except WebSocketDisconnect:
        await websocket.close()


@app.post("/uploader_")
async def uploader(request: Request, file_1: UploadFile = File(...)):
    #    try:
    img = Image.open(file_1.file)
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         content={'message': str(e)}
    #     )

    print("image's shape -- : ", np.array(img).shape)

    # delete all folders of runs/detect
    path_exp = f"{parent_path}/yolov5/runs/"
    _del = [rmtree(path) for path in Path(path_exp).glob("**/*")]

    # get the result of the model
    results = model(img)
    print("model finished ... ")

    # Results
    results.print()

    # save results
    results.save(save_dir=f"{parent_path}/yolov5/runs/detect/")

    return templates.TemplateResponse("second_page.html", context={"request": request})


@app.get("/download")
async def download(request: Request):
    # filename of the output file
    filename = "resultat_finale.jpg"

    # path to the result
    download_path = f"{parent_path}/yolov5/runs/detect/image0.jpg"

    return FileResponse(download_path, filename=filename)


@app.get("/Acceuil")
def Acceuil(request: Request):
    return templates.TemplateResponse("first_page.html", context={"request": request})


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, reload=True)
