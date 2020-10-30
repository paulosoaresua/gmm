import matplotlib.pyplot as plt
import matplotlib
import os

class TexFigure:
    """
    This class generates high-quality images from plots to be used in .tex files.
    """

    def __init__(self, output_dir, latex_doc_column_width=452.9679):
        """Parameters:
          - output_dir [string]: directory where images must be saved.
          - latex_doc_column_width [float]: width of the column in latex. Get this
                                            from LaTeX using \showthe\columnwidth
        """
        self.output_dir = output_dir
        self.latex_doc_column_width = latex_doc_column_width

    @staticmethod
    def configure_plots():
        params = {'backend': 'Agg',
                  'axes.labelsize': 10,
                  'font.size': 10}  # extend as needed
        matplotlib.rcParams.update(params)

    def save_image(self, filename):
        """Parameters:
          - filename [string]: name and extension of the image to be saved.
        """
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def new_figure(self):
        plt.figure(figsize=self._get_figsize(self.latex_doc_column_width, wf=1.0))

    def _get_figsize(self, column_width, wf=0.5, hf=(5. ** 0.5 - 1.0) / 2.0, ):
        """Parameters:
          - wf [float]:  width fraction in column_width units.
          - hf [float]:  height fraction in column_width units.
                         Set by default to golden ratio.
          - column_width [float]: width of the column in latex. Get this from LaTeX
                                  using \showthe\columnwidth
        Returns:  [fig_width,fig_height]: that should be given to matplotlib
        """
        fig_width_pt = column_width * wf
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        fig_width = fig_width_pt * inches_per_pt  # width in inches
        fig_height = fig_width * hf  # height in inches
        return [fig_width, fig_height]
