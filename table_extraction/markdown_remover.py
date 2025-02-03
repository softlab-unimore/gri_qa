from markdown import Markdown
from io import StringIO

class MarkdownRemover:
    def __init__(self):
        Markdown.output_formats["plain"] = self.unmark_element
        self.__md = Markdown(output_format="plain")
        self.__md.stripTopLevelTags = False

    def unmark_element(self, element, stream=None):
        if stream is None:
            stream = StringIO()
        if element.text:
            stream.write(element.text)
        for sub in element:
            self.unmark_element(sub, stream)
        if element.tail:
            stream.write(element.tail)
        return stream.getvalue()

    def unmark(self, text):
        return self.__md.convert(text)