from PIL import Image
import numpy as np
import cv2
import os
import glob


class Captcha(object):
    def __init__(self):
        self.input_path = 'data/input/input'
        self.output_path = 'data/output/output'
        self.train_path = 'data/train/'

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        pass

    def create_templates(self):
        for i in range(25):
            if i == 21:
                continue
            path = f"{self.input_path}{i:02}.jpg"
            chars = self._segment(path)
            with open(f"{self.output_path}{i:02}.txt", 'r') as file:
                truth = file.read().strip()
            assert len(chars) == len(truth)
            for j in range(len(chars)):
                cv2.imwrite(f'{self.train_path}{truth[j]}.jpg', chars[j])

    def load_templates(self):
        self.template = {}
        files = glob.glob(os.path.join(self.train_path, f'*.jpg'))
        for f in files:
            self.template[f.split('/')[-1].split('.')[0]] = self._read_template_images(f)

    def decode_images(self):
        for i in range(25):
            captcha = []
            if i == 21:
                continue
            path = f"{self.input_path}{i:02}.jpg"
            chars = self._segment(path)
            for c in chars:
                captcha.append(self._match_template(c))
            with open(f"{self.output_path}{i:02}.txt", 'r') as file:
                truth = file.read().strip()
                print(truth == ''.join(captcha))
            print(''.join(captcha), truth)

    def _match_template(self, char):
        best_match = None
        best_score = float('inf')  # Initialize with a large value
        for truth, value in self.template.items():
            if char.shape != value.shape:
                continue
            score = np.sum((char - value) ** 2)  # Mean squared error
            if score < best_score:
                best_score = score
                best_match = truth
        # print(best_match, score)
        return best_match

    def _read_template_images(self, path):
        binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _read_image(self, path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return binary

    def _segment(self, path):
        binary = self._read_image(path)
        chars = self._remove_margins_and_segment(binary)
        return chars

    def _find_margins(self, binary):
        return 11, 20, 5, 48

    def _remove_margins_and_segment(self, binary):
        top, bottom, left, right = self._find_margins(binary)
        binary = binary[top:bottom + 1, left:right + 1]
        character_width = 8
        characters = []
        for i in range(5):
            start_x = i * character_width + i
            end_x = start_x + character_width
            character_image = binary[:, start_x:end_x]
            characters.append(character_image)
        return characters


if __name__ == '__main__':
    # Captcha().create_templates()
    inst = Captcha()
    # inst.create_templates()
    inst.load_templates()
    inst.decode_images()