from kivy.app import App
from kivy.modules import inspector  # For inspection.
from kivy.core.window import Window  # For inspection.
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line


class DrawingWidget(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class BorderedLabel(Label):
    pass


class GUIApp(App):
    def build(self):
        inspector.create_inspector(Window, self)  # For inspection (press control-e to toggle).
        self.drawing_widget = DrawingWidget()

    def clear_canvas(self, obj):
        self.drawing_widget.canvas.clear()


if __name__ == '__main__':
    app = GUIApp()
    app.run()
