"""Read-only Dynamics current-versus-imagined window. Closing it never controls the robot."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    args=p.parse_args();title='Dynamics imagination / prediction is NOT execution'
    frame=np.zeros((398,640,3),np.uint8)
    cv2.putText(frame,'Waiting for Astra imagination...', (15,195),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
    path=args.session/'imagination/latest.json';last=None
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    try:
        while True:
            try:
                latest=json.loads(path.read_text())
                if latest['candidate_id']!=last:
                    im=cv2.imread(latest['panel'])
                    if im is not None:frame=im;last=latest['candidate_id']
            except (OSError,ValueError,KeyError):pass
            cv2.imshow(title,frame)
            if cv2.waitKey(100)&255 in (27,ord('q')):break
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
    finally:cv2.destroyAllWindows()

if __name__=='__main__':main()
