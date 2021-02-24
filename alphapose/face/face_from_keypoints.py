import numpy as np
import cv2
from alphapose.face.centerface import CenterFace
import pandas as pd
import time
class Face:
    def __init__(self):
        self.face_engine = CenterFace( landmarks=True)
        self.data = {}

    def clear_data(self):
        self.data = {}

    def export_data(self, fpath):
        pd.DataFrame(self.data).T.to_csv(fpath)

    def export_face_img(self, tracked_objects, orig_img, dir_path, vdo_fname='id'):
        is_finished = False
        rgb_img = orig_img[:, :, ::-1]
        [H, W, _] = rgb_img.shape
        self.face_engine.transform(orig_img.shape[0], orig_img.shape[1])
        face_dets, lms = self.face_engine(orig_img, threshold=0.35)

        for person in tracked_objects:

            # person is TrackedObject class in Norfair
            # Ref. https://github.com/tryolabs/norfair/blob/e062198487c12b32ca8b2197bc21227898d2dd31/norfair/tracker.py#L187
            keypoints = person.last_detection.points
            kp_score = person.last_detection.scores
            pid = person.id

            self.data[pid] = { 
                'info':'{}-{}'.format(pid, vdo_fname), 
                'found_face':False 
            }

            center_of_the_face = np.mean(keypoints[:5], axis=0)
            face_conf = np.mean(kp_score[:5], axis=0)

            face_keypoints = -1*np.ones((68,3))
            if face_conf > 0.5 and len(face_dets) > 0:
                face_min_dis = np.argmin(
                    np.sum(((face_dets[:, 2:4] + face_dets[:, :2]) / 2. - center_of_the_face) ** 2, axis=1))

                face_bbox = face_dets[face_min_dis][:4]
                face_prob = face_dets[face_min_dis][4]
                if center_of_the_face[0] < face_bbox[0] or center_of_the_face[1] < face_bbox[1] or center_of_the_face[0] > face_bbox[2] or center_of_the_face[1] > face_bbox[3]:
                    continue 
                if face_prob < 0.5:
                    continue

                face_image = orig_img[int(face_bbox[1]): int(face_bbox[3]), int(face_bbox[0]): int(face_bbox[2])]

                cv2.imwrite('{}/{}-{}.jpg'.format(dir_path, pid, time.time()), face_image)
                is_finished = True
                self.data[pid]['found_face'] = True
        
        return is_finished