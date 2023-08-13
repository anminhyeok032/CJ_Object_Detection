import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pandas as pd
import openpyxl #패키지 불러오기

labels = pd.read_excel('/content/yolov7/labels.xlsx',index_col='번호') # 라벨링 파일 가져오기

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # IoU 계산 함수
    def ioU(box1,box2):
      box1_area = (int(box1[3]) - int(box1[1]) + 1) * (int(box1[4]) - int(box1[2]) + 1)
      box2_area = (int(box2[3]) - int(box2[1]) + 1) * (int(box2[4]) - int(box2[2]) + 1)
                    
      x1 = max(int(box1[1]),int(box2[1]))
      y1 = max(int(box1[2]),int(box2[2]))
      x2 = min(int(box1[3]),int(box2[3]))
      y2 = min(int(box1[4]),int(box2[4]))
                    
      w = max(0, x2 - x1 + 1)
      h = max(0, y2 - y1 + 1)
      inter = w * h
      iou = inter / (box1_area + box2_area - inter)
      return iou

    # 제공된 좌표 값의 라벨이 인덱스 일 때
    def ioU_index(ls_temp, im0):
      filename = "/content/yolov7/iou_index.txt" #파일명
      filename_check = filename.split('.')
      
      if filename_check[1] == "xlsx":   # 엑셀 파일일 때
        book = openpyxl.load_workbook(filename)
        sheet=book.worksheets[0]
        iou_coordinates = [] #리스트 자료형 생성

        for i, row in enumerate(sheet.rows): #전체 행에 대하여 반복실행
          # 만약 좌표가 x1y1, x2y2 기준으로 제공될 경우 아래부분은 주석후 사용할 것
          # ***왼쪽 위와 오른쪽 아래 꼭짓점 기준***
          iou_coordinates.append([
              row[0].value, #1열 데이터
              row[1].value, #2열 데이터
              row[2].value, #3열 데이터
              row[3].value, #4열 데이터
              row[4].value, #5열 데이터
          ])

          # ***왼쪽 위 꼭짓점 기준***
          # x,y,w,h, 기준 제공될 경우
          # if i != 0:   # ***첫 줄에 아무표시 없을시 삭제***
          #   iou_coordinates[i][3] = iou_coordinates[i][1] + iou_coordinates[i][3] # width
          #   iou_coordinates[i][4] = iou_coordinates[i][2] + iou_coordinates[i][4] # height

        iou_coordinates.pop(0)  # ***첫 줄에 아무표시 없을시 삭제***

      elif filename_check[1] == "txt":    # txt 파일일 때
        with open(filename, 'r') as file:
          lines = file.readlines()

        # 문자열 처리 및 리스트에 저장
        iou_coordinates = [] # 리스트 자료형 생성
        for i,line in enumerate(lines):
          line = line.replace(',', '').replace('"', '').replace("'", "")
          values = line.strip().split()  # 띄어쓰기 기준으로 문자열 분리하여 리스트로 변환
          iou_coordinates.append(values)

          # ***왼쪽 위 꼭짓점 기준***
          # x,y,w,h, 기준 제공될 경우
          # if i != 0:   # ***첫 줄에 아무표시 없을시 삭제***
          #   iou_coordinates[i][3] = int(iou_coordinates[i][1]) + int(iou_coordinates[i][3]) # width
          #   iou_coordinates[i][4] = int(iou_coordinates[i][2]) + int(iou_coordinates[i][4]) # height
        iou_coordinates.pop(0)  # ***첫 줄에 아무표시 없을시 삭제***
        print(iou_coordinates)

      print('*** iou 계산을 위해 제공된 좌표 값 ***')
      str_iou_coordinates= []
      for iou_coordinate in iou_coordinates :
        str_iou_coordinate = [str(num) for num in iou_coordinate]
        str_iou_coordinates.append(str_iou_coordinate)
      print(str_iou_coordinates)

      print('*** 인식된 상품 좌표 값 ***')
      print(ls_temp)

      productList = []
      for str_iou_coordinate in str_iou_coordinates :
        temp_iou = []
        for i in range(0,len(ls_temp)):
          if str_iou_coordinate[0] == ls_temp[i][0]:
            temp_iou.append(round(ioU(str_iou_coordinate,ls_temp[i]),2))

        # 사각형 그리기
        x1, y1, x2, y2 = map(int, str_iou_coordinate[1:5])
        label = labels.loc[int(str_iou_coordinate[0])]["이름"]
        im0 = plot_one_box([x1, y1, x2, y2], im0, label=None, color=(0, 0, 255), line_thickness=1)  # 빨간색으로 기존 물체 bounding box

        print(f'{str_iou_coordinates.index(str_iou_coordinate)+1} / {len(str_iou_coordinates)} 결과')    
        if temp_iou:
          print(temp_iou)
          print([labels.loc[int(str_iou_coordinate[0])]["이름"],max(temp_iou)])    
          productList.append([labels.loc[int(str_iou_coordinate[0])]["이름"],max(temp_iou)])
        else:
          print("*** 같은 클래스의 Object가 없습니다! ***")

      # im0 = annotator.result()  

      f=open('cjproject_result.txt','w')
      for product in productList:
          data=f'{product[0]} {product[1]}\n'
          f.write(data)
      f.close()
      print('*** 데이터 저장 완료 ***\n')
      print('result : cjproject_result.txt')

      return im0



    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # annotator = Annotator(im0, line_width=line_thickness, example=str('박스이름'))
            
            ls_temp=[] # 감지된 바운딩 박스의 좌표 값을 IoU계산을 위해 잠시 담아두는 리스트.
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        line = (cls, *xyxy, conf) if opt.save_conf else (cls, *xyxy)  # label format

                        temp_line = (('%g ' * len(line)).rstrip() % line + '\n')
                        coordinate = temp_line.split()
                        # IoU 계산을 위해 잠시 리스트에 담기
                        coordinate[0] = names[int(cls)]
                        ls_temp.append(coordinate)
                        print(ls_temp)

                        label = f'{(labels.loc[int(names[int(cls)])]["이름"])} {conf:.2f}'                  
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            for i in range(0,1):
              if view_img:
                  cv2.imshow(str(p), im0)
                  cv2.waitKey(1)  # 1 millisecond
              im0 = ioU_index(ls_temp, im0) # 처음 박스를 그리고 위에 기존 박스 위치 그려줌

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
