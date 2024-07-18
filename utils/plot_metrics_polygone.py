import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from typing import Tuple

map_metric_names = {
    "FID": "FID",
    "MMS": "MMS",
    "COV": "Coverage",
    "Density": "Density",
    "APD": "APD"
}

def _register_radar_projection(numberOfMetrics: int = None, frame: str = "polygon"):
    angles = np.linspace(0, 2 * np.pi, numberOfMetrics, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(numberOfMetrics)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "polygon-chart"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(angles), labels)

        def _gen_axes_patch(self):
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), numberOfMetrics, radius=0.5, edgecolor="k")

        def draw(self, renderer):
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = numberOfMetrics
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(numberOfMetrics),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return angles

def _normalize_metrics(df_metrics: pd.DataFrame):
    on_column = df_metrics['ON']
    metrics_columns = df_metrics.drop('ON', axis=1)
    normalized_metrics = metrics_columns.divide(metrics_columns.max())
    normalized_dataframe = pd.DataFrame(normalized_metrics, columns=metrics_columns.columns)
    return pd.concat([on_column, normalized_dataframe], axis=1)

def _transform_metrics(df_metrics, usedMetrics):
    df_copy = df_metrics.copy()
    on_names = list(df_metrics["ON"])
    reference = on_names[0]  # Use the first entry as the reference

    for _metric in usedMetrics:
        df_metric_real = df_copy.loc[df_copy["ON"] == reference]
        _metric_real = df_metric_real[_metric].iloc[0]

        for on_name in on_names:
            df_metric_row = df_copy.loc[df_copy["ON"] == on_name]
            _metric_model = df_metric_row[_metric].iloc[0]

            if "fid" in _metric.lower():
                if _metric_model > _metric_real:
                    df_metrics.loc[df_metrics["ON"] == on_name, _metric] = 1.0 - (_metric_model - _metric_real)
                else:
                    df_metrics.loc[df_metrics["ON"] == on_name, _metric] = 1.0 + (_metric_real - _metric_model)
            else:
                if _metric_model < _metric_real:
                    df_metrics.loc[df_metrics["ON"] == on_name, _metric] = 1.0 - (_metric_real - _metric_model)
                else:
                    df_metrics.loc[df_metrics["ON"] == on_name, _metric] = 1.0 + (_metric_model - _metric_real)

        df_metrics.loc[df_metrics["ON"] == reference, _metric] = 1.0

    return df_metrics

def plot_metrics_on_polygone(
    df_metrics,
    usedMetrics: list = None,
    frame: str = "polygon",
    title: str = None,
    figsize: Tuple[int, int] = (5, 5),
):
    df_metrics = _normalize_metrics(df_metrics=df_metrics)

    if usedMetrics is None:
        metrics_ = list(df_metrics.columns)
        metrics_.remove('ON')
        metrics = [_metric for _metric in metrics_ if not "std" in _metric]
    else:
        metrics = usedMetrics

    numberOfMetrics = len(metrics)
    df_metrics = _transform_metrics(df_metrics=df_metrics, usedMetrics=metrics)

    angles = _register_radar_projection(numberOfMetrics=numberOfMetrics, frame=frame)

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, subplot_kw=dict(projection="polygon-chart"))

    colors = plt.get_cmap("Dark2")(np.linspace(start=0.0, stop=1.0, num=len(df_metrics['ON'].unique())))

    ax.set_rgrids(np.linspace(start=0.1, stop=df_metrics.select_dtypes(include=["number"]).max().max(), num=5))
    ax.set_title(title, weight="bold", size="medium", position=(0.5, 1.1), horizontalalignment="center", verticalalignment="center")

    for i, modelName in enumerate(df_metrics['ON'].unique()):
        color = colors[i]
        metricData = df_metrics.loc[df_metrics["ON"] == modelName][metrics]
        _metricData = [float(metricData[_metric].iloc[0]) for _metric in metrics]
        ax.plot(angles, _metricData, color=color, label=modelName)
        ax.fill(angles, _metricData, facecolor=color, alpha=0.25, label="_nolegend_")

    ax.set_varlabels([map_metric_names[_metric] for _metric in metrics])
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.15))
    fig.savefig(title + ".png")
    plt.close(fig)  # Close the figure after saving to free up memory

def process_and_plot_csv(file_path: str, title: str):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        plot_metrics_on_polygone(df_metrics=df, title=title)
if __name__ == "__main__":
    models=['ASCVAE','SVAE']
    reg=['REG','STGCN']
    for model in models :
        output_directory_results = f'../results/Generative_models/Score+Action_conditioned/{model}/Wrec_0.999_Wkl_0.001/'
        for run in range(5):
            print('run------',run)
            for i in range(5):
                print('class------',i)
                for m in reg:
                    class_directory = os.path.join(output_directory_results, f'run_{run}/class_{i}')
                    csv_files = [
                        (os.path.join(class_directory, f'{m}_train_vs_noisy_vs_gen_{i}.csv'), f'{m}_polygone_train_vs_noisy_vs_gen_class_{i}'),
                        (os.path.join(class_directory, f'{m}_train_vs_test{i}.csv'), f'{m}_polygone_train_vs_test_vs_gen_class_{i}')]
                    for csv_file, title in csv_files:
                        process_and_plot_csv(file_path=csv_file, title=os.path.join(class_directory, title))
                    





