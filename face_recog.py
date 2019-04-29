from face_module.config import get_config
from face_module.mtcnn import MTCNN
from face_module.utils import prepare_facebank, load_facebank
from face_module.Learner import face_learner
from PIL import Image
import numpy as np


class FaceRecog:
    """
        This Module checks the person if the given person is in the facebank.
    """

    def __init__(self):
        self.conf = get_config(False)
        self.mtcnn = MTCNN()
        self.learner = face_learner(self.conf, True)
        self.targets, self.names = (None, None)
        self.colors = None

    def build(self, threshold, face_embeddings):
        self.learner.threshold = threshold
        self.learner.load_state(self.conf, 'ir_se50.pth', True, True)
        if face_embeddings:
            self.targets, self.names = prepare_facebank(self.conf, self.learner.model, self.mtcnn)
        else:
            self.targets, self.names = load_facebank(self.conf)
        self.colors = np.random.randint(0, 255, size=(len(self.names), 3), dtype="uint8")

    def find_face(self, frame, x, y, w, h):
        # crop the person in the bounding box.
        image = Image.fromarray(frame)
        image = image.crop((x, y, x + w, y + h))
        # image.save('out' + str(x) + '.png')
        try:
            bbox, face = self.mtcnn.align(image)
        except:
            bbox = []
            face = []
        # identify the person. If identified return the name else return person
        if len(bbox) == 0:
            color = [int(c) for c in self.colors[0]]
            return 'person', color
        else:
            results, score = self.learner.infer(self.conf, [face], self.targets, True)
            color = [int(c) for c in self.colors[results[0] + 1]]
            return self.names[results[0] + 1], color
