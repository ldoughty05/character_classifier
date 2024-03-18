import glob
import os
import random

import cv2
from kivy.app import App
from kivy.core.window import Window  # For inspection.
from kivy.graphics import Color, Line
from kivy.modules import inspector  # For inspection.
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.stencilview import StencilView

import classifier
from classifier import accessor


class DrawingWidget(StencilView):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class BorderedLabel(Label):
    pass


class CharacterGuess(BoxLayout):
    character = StringProperty()
    probability = StringProperty()

    def update(self, probability, character):
        self.character = character
        self.probability = str(probability // 0.01)


class ImageSpace(Image):
    def set_image_from_index(self, index):
        try:
            self.source = f"test_image_{index}.png"
            self.reload()
        except FileNotFoundError:
            self.source = "placeholder.png"
            self.reload()


def get_model_list():
    models = list()
    for file in glob.glob('*.pth'):
        models.append(file)
    return models


class GUIApp(App):
    model_name = StringProperty(accessor['model_name'])
    dataset = StringProperty(str(accessor['dataset']))
    device = StringProperty(accessor['device'])
    accuracy = StringProperty(str(accessor['accuracy']))
    loss = StringProperty(str(accessor['loss']))
    trainable_params = StringProperty(str(accessor['trainable_params']))
    sidebar_items = list()  # what are the benefits of turning a list into a tuple?
    num_test_images = 0
    current_image = "placeholder.png"
    previous_test_image_spaces = list()
    model_list = get_model_list()
    active_model_name = None
    active_dataset = None

    def build(self):
        for i in range(10):
            character_guess_object = CharacterGuess(probability='0.00', character='-')
            self.root.ids.sidebar.add_widget(character_guess_object)
            self.sidebar_items.append(character_guess_object)
        self.evaluate(self.model_list[0])  # assumes there is at least one element.
        # TODO: handle the case where there is less than one element
        for j in range(10):
            image_space_object = ImageSpace()
            self.root.ids.previous_test_image_space_grid.add_widget(image_space_object)
            self.previous_test_image_spaces.append(image_space_object)
        self.root.ids.model_selector.values = self.model_list
        inspector.create_inspector(Window, self)  # For inspection (press control-e to toggle).

    def clear_canvas(self):
        drawing_widget = self.root.ids.drawing_widget
        drawing_widget.canvas.clear()

    def test_random_image(self):
        random_image_index = random.randint(0, 1000)
        label = str(classifier.get_test_image_label(random_image_index, self.active_dataset))
        predictions = classifier.classify_training_image_index(random_image_index, self.active_dataset, self.active_model_name,
                                                               self.get_next_current_image_filename())
        self.update_sidebar(predictions)
        self.increment_current_image_index()
        self.root.ids.test_random_image_label.text = f"Label: {label}"

    def test_drawing_with_model(self):
        formatted_image = self.drawing_widget_to_png()
        # formatted_image is 'test_image_#.png'
        predictions = classifier.classify_image(formatted_image, self.active_dataset, self.active_model_name)
        self.update_sidebar(predictions)
        self.increment_current_image_index()
        self.root.ids.test_random_image_label.text = f"Label: hand drawn"

    def drawing_widget_to_png(self):
        # Get drawing from widget
        drawing_widget = self.root.ids.drawing_widget
        file = f"user_made_char.png"
        drawing_widget.export_to_png(file)
        # format the widget png to a 28x28 greyscale
        test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
        formatted_image = self.get_next_current_image_filename()
        cv2.imwrite(formatted_image, image_resized)
        return formatted_image  # returns the filepath, not the image itself

    def update_sidebar(self, predictions):
        for i in range(len(self.sidebar_items)):
            probability, character = predictions[i]
            self.sidebar_items[i].update(probability, character)

    def update_variables(self):
        self.model_name = accessor['model_name']
        self.dataset = accessor['dataset']
        self.device = accessor['device']
        self.accuracy = f"{accessor['accuracy']:>0.2f}"
        self.loss = f"{accessor['loss']:>7f}"
        self.trainable_params = str(accessor['trainable_params'])

    def get_next_current_image_filename(self):
        filename = f"test_image_{self.num_test_images}.png"
        self.num_test_images += 1
        return filename

    def increment_current_image_index(self):
        self.root.ids.current_image.source = f"test_image_{self.num_test_images - 1}.png"
        items_in_grid = 9
        for i, image_space in enumerate(self.previous_test_image_spaces):
            image_space.set_image_from_index(self.num_test_images - 1 - i)
        # make this delete unused files

    def evaluate(self, model_name):
        classifier.evaluate(model_name)
        self.active_model_name = model_name
        self.active_dataset = model_name.split("_")[0]
        print(accessor['accuracy'])
        self.update_variables()

    def model_selected(self, model_name):
        self.evaluate(model_name)


if __name__ == '__main__':
    for old_test_image_file in glob.glob("./test_image_*"):
        os.remove(old_test_image_file)
    app = GUIApp()
    app.run()
