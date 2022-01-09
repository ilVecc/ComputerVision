from pathlib import Path
import cv2.cv2 as cv

folder_src = Path('../imgs/colosseum/low_res')
folder_dst = Path('../imgs/colosseum/low_res')
folder_dst.mkdir(exist_ok=True)

if __name__ == '__main__':
    
    scale = 4
    
    for img_path in folder_src.iterdir():
        img = cv.imread(str(img_path))
        img_out = cv.resize(img, (3840 // scale, 2880 // scale))
        img_out_path = folder_dst / img_path.name
        cv.imwrite(str(img_out_path), img_out)
