from face_module.config import get_config
from face_module.mtcnn import MTCNN
from face_module.utils import prepare_facebank, load_facebank
from face_module.Learner import face_learner
from PIL import Image


class FaceRecog:
    """
        This Module checks the person if the given person is in the facebank.
    """

    def __init__(self):
        self.conf = get_config(False)
        self.mtcnn = MTCNN()
        self.learner = face_learner(self.conf, True)

    def build(self, threshold):
        self.learner.threshold = threshold
        self.learner.load_state(self.conf, 'ir_se50.pth', True, True)
        self.targets, self.names = prepare_facebank(self.conf, self.learner.model, self.mtcnn)

    def find_face(self, frame, x, y, w, h):
        # crop the person in the bounding box.
        image = Image.fromarray(frame)
        image = image.crop((x, y, x + w, y + h))

        try:
            bbox, face = self.mtcnn.align(image)
        except:
            bbox = []
            face = []
        # identify the person. If identified return the name else return person
        if len(bbox) == 0:
            return 'person'
        else:
            results, score = self.learner.infer(self.conf, [face], self.targets, True)
            print(results[0])
            return self.names[results[0]]
