import subprocess
import unittest
import os


class Doc_Test(unittest.TestCase):

    def setUp(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        self.path_to_docs = os.path.sep.join(
            dirname.split(os.path.sep)[:-1] + ['docs']
        )
        self.doctrees_path = os.path.sep.join(
            self.path_to_docs.split(os.path.sep) + ['_build', 'doctrees']
        )
        self.html_path = os.path.sep.join(
            self.path_to_docs.split(os.path.sep) + ['_build', 'html']
        )
        self.latex_path = os.path.sep.join(
            self.path_to_docs.split(os.path.sep) + ['_build', 'latex']
        )
        self.link_path = os.path.sep.join(
            self.path_to_docs.split(os.path.sep) + ['_build']
        )

        print self.path_to_docs

    def test_html(self):
        check = subprocess.call([
            "sphinx-build", "-nW", "-b", "html", "-d",
            "{0!s}".format(self.doctrees_path) ,
            "{0!s}".format(self.path_to_docs),
            "{0!s}".format(self.html_path)
        ])
        assert check == 0

    def test_latex(self):
        check = subprocess.call([
            "sphinx-build", "-nW", "-b", "latex", "-d",
            "{0!s}".format(self.doctrees_path),
            "{0!s}".format(self.path_to_docs),
            "{0!s}".format(self.latex_path)])
        assert check == 0

    def test_linkcheck(self):
        check = subprocess.call([
            "sphinx-build", "-nW", "-b", "linkcheck", "-d",
            "{0!s}".format(self.doctrees_path),
            "{0!s}".format(self.path_to_docs),
            "{0!s}".format(self.link_path)])
        assert check == 0

if __name__ == '__main__':
    unittest.main()
