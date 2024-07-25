from PIL import Image
import numpy as np
import cv2
import os
import glob


class Captcha:
    def __init__(self, mode):
        # mode = 'full': train & predict
        # mode = 'predict': predict only
        self.input_path = 'data/input/input'
        self.output_path = 'data/output/output'
        self.train_path = 'data/train/'
        self.num_images = 25
        self.images = []
        self.truths = []
        self.fused_image = np.zeros((30, 60), dtype=np.uint8)
        self.top_index = None if mode != 'predict' else 11
        self.bottom_index = None if mode != 'predict' else 20
        self.left_index = None if mode != 'predict' else 5
        self.right_index = None if mode != 'predict' else 48
        if mode == 'full':
            self.train()

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        self.predict(im_path, save_path)

    def train(self):
        self._load_all_images()
        self._infer_margin()
        for i in range(len(self.images)):
            chars = self._remove_margins_and_segment(self.images[i])
            for j in range(len(chars)):
                # store each character's image as photo for prediction/inference phase
                cv2.imwrite(f'{self.train_path}{self.truths[i][j]}.jpg', chars[j])

    def predict(self, im_path, save_path):
        self._load_templates()
        self._decode_image(im_path, save_path)

    def _load_all_images(self):
        """
        load all input images and truth texts
        :return: None
        """
        for i in range(self.num_images):
            if i == 21:
                continue
            path = f"{self.input_path}{i:02}.jpg"
            self.images.append(self._read_image(path))
            with open(f"{self.output_path}{i:02}.txt", 'r') as file:
                self.truths.append(file.read().strip())

    def _infer_margin(self):
        """
        1. detect the non-character areas and return the margin of 4 directions
        2. This helps crop the image and split into 5 parts
        :return:
        """
        for img in self.images:
            self.fused_image = cv2.bitwise_or(self.fused_image, img)
        # Get the dimensions of the image
        height, width = self.fused_image.shape

        # Find top margin index
        for row in range(height):
            if np.any(self.fused_image[row, :] == 255):
                self.top_index = row
                break
        # Find bottom margin index
        for row in range(height - 1, -1, -1):
            if np.any(self.fused_image[row, :] == 255):
                self.bottom_index = row
                break
        # Find left margin index
        for col in range(width):
            if np.any(self.fused_image[:, col] == 255):
                self.left_index = col
                break
        # Find right margin index
        for col in range(width - 1, -1, -1):
            if np.any(self.fused_image[:, col] == 255):
                self.right_index = col
                break

    def _read_image(self, path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return binary

    def _read_template_images(self, path):
        binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _remove_margins_and_segment(self, binary):
        """
        1. remove margins, whose parameters is based on _infer_margin method
        2. split the cropped image into 5 parts, each with precisely one character
        :param binary: input image
        :return:
        """
        binary = binary[self.top_index:self.bottom_index + 1, self.left_index:self.right_index + 1]
        character_width = 8
        characters = []
        for i in range(5):
            start_x = i * character_width + i
            end_x = start_x + character_width
            character_image = binary[:, start_x:end_x]
            characters.append(character_image)
        return characters

    def _load_templates(self):
        """
        load each character image into a dictionary
        :return:
        """
        self.template = {}
        files = glob.glob(os.path.join(self.train_path, f'*.jpg'))
        for f in files:
            self.template[f.split('/')[-1].split('.')[0]] = self._read_template_images(f)

    def _decode_image(self, im_path, save_path):
        captcha = ''
        binary = self._read_image(im_path)
        chars = self._remove_margins_and_segment(binary)
        for c in chars:
            captcha += self._match_template(c)
        print('The decoded captcha is: ', captcha)
        with open(save_path, "w") as file:
            # Write the string to the file
            file.write(captcha)

    def _match_template(self, char):
        """
        1. find out which char in templates is closest to the char under examination
        2. use mean square error as metric
        :param char:
        :return: best_match: the most likely char
        """
        best_match = None
        best_score = float('inf')  # Initialize with a large value
        for truth, value in self.template.items():
            if char.shape != value.shape:
                continue
            score = np.sum((char - value) ** 2)  # Mean squared error
            if score < best_score:
                best_score = score
                best_match = truth
        return best_match


if __name__ == '__main__':
    instance = Captcha('full')
    for i in range(20):
        instance(f'data/input/input{i:02}.jpg', 'data/result/result.txt')

