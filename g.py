import json
from random import sample
import matplotlib.patches as mpatches
import pandas as pd
import qdarktheme

import matplotlib
import matplotlib.pyplot as plt
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import TimeDelta, Time

matplotlib.use('Qt5Agg')
plt.style.use('dark_background')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from octans import XLightCurve
from PyQt5 import QtWidgets, QtGui
from PyQt5 import QtCore
from sys import argv

from gui import main, file_loader, graph, savitzky_golay, b_spline, savgol_filter, butterworth_filter, add_flux, \
    add_time, time_travel, fold_periodogram, fold_phase, minimum, about

DARK_COLORS = [
    "#FFEB3B", "#9C27B0", "#00BCD4", "#FF9800", "#8BC34A",
    "#FF4081", "#4CAF50", "#FF5722", "#03A9F4", "#E91E63",
    "#CDDC39", "#673AB7", "#2196F3", "#FFC107", "#9E9E9E",
    "#FF5252", "#FFEB3B", "#4CAF50", "#03A9F4", "#E91E63",
    "#9C27B0", "#00BCD4", "#FF9800", "#8BC34A", "#FF4081",
    "#FF5722", "#4CAF50", "#03A9F4", "#CDDC39", "#673AB7",
    "#2196F3", "#FFC107", "#9E9E9E", "#FF5252", "#FFEB3B",
    "#4CAF50", "#03A9F4", "#E91E63", "#9C27B0", "#00BCD4",
    "#FF9800", "#8BC34A", "#FF4081", "#FF5722", "#4CAF50",
    "#03A9F4", "#CDDC39", "#673AB7", "#2196F3", "#FFC107"
]


class MainWindow(QtWidgets.QMainWindow, main.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)
        self.flags = QtCore.Qt.Window
        self.setWindowFlags(self.flags)

        self.setWindowIcon(QtGui.QIcon('xtrema31.png'))

        self.loader_form = None
        self.xlc = []

        self.treeWidget.installEventFilter(self)

        self.actionLoad.triggered.connect(lambda: (self.open_file_loader()))
        self.actionPlot.triggered.connect(lambda: (self.plot_light_curve()))
        self.actionQuit.triggered.connect(lambda: (self.close()))
        self.actionAbout.triggered.connect(lambda: (self.about()))

    def about(self):
        about_to_show = AboutForm(self)
        self.mdiArea.addSubWindow(about_to_show)
        about_to_show.show()

    def minimum(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        minimum_to_show = MinimumForm(self, each)
                        self.mdiArea.addSubWindow(minimum_to_show)
                        minimum_to_show.show()

    def phase_fold(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        phase_fold_to_show = PhaseFoldForm(self, each)
                        self.mdiArea.addSubWindow(phase_fold_to_show)
                        phase_fold_to_show.show()

    def periodogram_fold(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        periodogram_fold_to_show = PeriodogramFoldForm(self, each)
                        self.mdiArea.addSubWindow(periodogram_fold_to_show)
                        periodogram_fold_to_show.show()

    def normalize(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        normalize_to_show = NormalizeForm(self, each)
                        self.mdiArea.addSubWindow(normalize_to_show)
                        normalize_to_show.show()

    def periodogram(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        periodogram_to_show = PeriodogramForm(self, each)
                        self.mdiArea.addSubWindow(periodogram_to_show)
                        periodogram_to_show.show()

    def time_travel_calculate(self, time_travel_type):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        time_travel_to_show = TimeTravelForm(self, each, item, time_travel_type)
                        self.mdiArea.addSubWindow(time_travel_to_show)
                        time_travel_to_show.show()

    def add_time(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        add_time_to_show = AddTimeForm(self, each, item)
                        self.mdiArea.addSubWindow(add_time_to_show)
                        add_time_to_show.show()

    def add_flux(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        add_flux_to_show = AddFluxForm(self, each, item)
                        self.mdiArea.addSubWindow(add_flux_to_show)
                        add_flux_to_show.show()

    def boundaries(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        boundaries_to_show = BoundariesForm(self, each)
                        self.mdiArea.addSubWindow(boundaries_to_show)
                        boundaries_to_show.show()

    def smooth_savitzky_golay(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        savitzky_golay_to_show = SavitzkyGolayForm(self, each)
                        self.mdiArea.addSubWindow(savitzky_golay_to_show)
                        savitzky_golay_to_show.show()

    def smooth_b_spline(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        b_spline_to_show = BSplineForm(self, each)
                        self.mdiArea.addSubWindow(b_spline_to_show)
                        b_spline_to_show.show()

    def smooth_savgol_filter(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        savgol_filter_to_show = SavgolFilterForm(self, each)
                        self.mdiArea.addSubWindow(savgol_filter_to_show)
                        savgol_filter_to_show.show()

    def smooth_butterworth_filter(self):
        selected = self.treeWidget.selectedItems()
        if selected:
            item = self.treeWidget.selectedItems()[0]
            if item.parent() is None:
                for each in self.xlc:
                    if str(each) == item.text(0):
                        butterworth_filter_to_show = ButterworthFilterForm(self, each)
                        self.mdiArea.addSubWindow(butterworth_filter_to_show)
                        butterworth_filter_to_show.show()

    def delete(self):
        for item in self.treeWidget.selectedItems():
            if item.parent():
                the_row = self.treeWidget.indexFromItem(item).row()
                for each in self.xlc:
                    if str(each) == item.parent().text(0):
                        del each[the_row]
                item.parent().removeChild(item)
            else:
                del self.xlc[self.treeWidget.indexOfTopLevelItem(item)]
                self.treeWidget.takeTopLevelItem(self.treeWidget.indexOfTopLevelItem(item))

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.ContextMenu and source is self.treeWidget:
            active = len(self.treeWidget.selectedItems()) > 0
            menu = QtWidgets.QMenu()
            menu.addAction('Load...', lambda: (self.open_file_loader()))
            delete = menu.addAction('Delete', lambda: (self.delete()))
            delete.setEnabled(active)
            menu.addSeparator()
            analysis_menu = menu.addMenu("Analysis")
            analysis_menu.setEnabled(active)

            analysis_menu.addAction('Plot...', lambda: (self.plot_light_curve()))

            smoothify_menu = analysis_menu.addMenu("Smoothify")
            smoothify_menu.addAction("Savitzky Golay...", lambda: (self.smooth_savitzky_golay()))
            smoothify_menu.addAction("B Spline...", lambda: (self.smooth_b_spline()))
            smoothify_menu.addAction("Savgol Filter...", lambda: (self.smooth_savgol_filter()))
            smoothify_menu.addAction("Butterworth Filter...", lambda: (self.smooth_butterworth_filter()))

            boundary_menu = analysis_menu.addMenu('Boundary')
            boundary_menu.addAction("Extrema...", lambda: (self.boundaries()))

            analysis_menu.addAction('Minima...', lambda: (self.minimum()))

            flux_menu = analysis_menu.addMenu('Flux')
            flux_menu.addAction("Add...", lambda: (self.add_flux()))

            time_menu = analysis_menu.addMenu('Time')
            time_menu.addAction("Add...", lambda: (self.add_time()))
            time_menu.addAction("Heliocentric JD...", lambda: (self.time_travel_calculate("hjd")))
            time_menu.addAction("Barycentric JD...", lambda: (self.time_travel_calculate("bjd")))

            normalization_menu = analysis_menu.addMenu('Normalization')
            normalization_menu.addAction("Normalize...", lambda: (self.normalize()))
            normalization_menu.addAction("Periodogram...", lambda: (self.periodogram()))

            fold_menu = analysis_menu.addMenu('Fold')
            fold_menu.addAction("Periodogram...", lambda: (self.periodogram_fold()))
            fold_menu.addAction("Epoch...", lambda: (self.phase_fold()))

            menu.exec_(event.globalPos())
            return True

        return super(MainWindow, self).eventFilter(source, event)

    def plot_light_curve(self):
        times, fluxes, flux_errs = [], [], []
        for item in self.treeWidget.selectedItems():
            if not item.parent():
                for each in self.xlc:
                    if str(each) == item.text(0):
                        graph_to_show = GraphForm(self, lambda ax: each.plot(ax=ax))
                        self.mdiArea.addSubWindow(graph_to_show)
                        graph_to_show.show()
            else:

                the_row = self.treeWidget.indexFromItem(item).row()
                for each in self.xlc:
                    if str(each) == item.parent().text(0):
                        times.append(each.time[the_row].jd)
                        fluxes.append(each.flux[the_row].value)
                        flux_errs.append(each.flux_err[the_row].value)

        if times and fluxes and flux_errs:
            xlc = XLightCurve(times, fluxes, flux_errs)
            graph_to_show = GraphForm(self, lambda ax: xlc.plot(ax=ax))
            self.mdiArea.addSubWindow(graph_to_show)
            graph_to_show.show()

    def open_file_loader(self):
        if self.loader_form is None:
            self.loader_form = LoadForm(parent=self)
            self.mdiArea.addSubWindow(self.loader_form)
            self.loader_form.show()
        else:
            self.loader_form.raise_()


class AboutForm(QtWidgets.QWidget, about.Ui_Dialog):
    def __init__(self, parent):
        super(AboutForm, self).__init__(parent)
        self.setupUi(self)

        pix = QtGui.QPixmap("xtrema31.png")
        self.pixmap = pix.scaled(128, 128, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                 QtCore.Qt.TransformationMode.FastTransformation)

        self.label.setPixmap(self.pixmap)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, width, height, dpi):
        self.fig = Figure(tight_layout=True, figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class GraphForm(QtWidgets.QWidget, graph.Ui_Form):
    def __init__(self, parent, func):
        super(GraphForm, self).__init__(parent)
        self.setupUi(self)

        sc = MplCanvas(5, 4, 100)
        func(sc.axes)
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.addWidget(widget)
        self.show()


class BoundariesForm(QtWidgets.QWidget, graph.Ui_Form):
    def __init__(self, parent, xlc):
        super(BoundariesForm, self).__init__(parent)
        self.setupUi(self)

        self.xlc = xlc

        self.sc = MplCanvas(5, 4, 100)
        self.xlc.plot(ax=self.sc.axes)
        toolbar = NavigationToolbar(self.sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)

        self.sigma_multiplier = QtWidgets.QDoubleSpinBox()
        self.sigma_multiplier.setMaximum(9999.0)
        self.sigma_multiplier.setProperty("value", 1.0)
        self.sigma_multiplier.setMinimum(0.2)

        calculate = QtWidgets.QPushButton()
        calculate.setText("Calculate")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        calculate.clicked.connect(lambda: (self.calculate()))

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.addWidget(widget, 0, 0, 1, 4)
        self.gridLayout.addWidget(self.sigma_multiplier, 1, 0, 1, 3)
        self.gridLayout.addWidget(calculate, 1, 3, 1, 1)
        self.show()

    def calculate(self):
        try:
            boundaries = self.xlc.boundaries_extrema(sigma_multiplier=self.sigma_multiplier.value())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        for line in self.sc.axes.lines[1:]:
            line.remove()

        for boundary in boundaries:
            self.sc.axes.axvline(x=boundary[0], color="b")
            self.sc.axes.axvline(x=boundary[1], color="y")
        self.sc.draw()


class MinimumForm(QtWidgets.QWidget, minimum.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(MinimumForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.frame.close()
        self.sc = MplCanvas(5, 4, 100)
        toolbar = NavigationToolbar(self.sc, self)
        self.xlc.plot(ax=self.sc.axes)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)

        self.gridLayout_3.addLayout(layout, 0, 0, 1, 2)
        self.left = None
        self.right = None

        self.treeWidget.installEventFilter(self)
        self.tableWidget.installEventFilter(self)

        self.tableWidget.itemSelectionChanged.connect(lambda: (self.highlight_selected()))
        self.pushButton.clicked.connect(lambda: (self.calculate()))
        self.pushButton_2.clicked.connect(lambda: (self.find_boundaries()))
        self.sc.fig.canvas.mpl_connect("button_press_event", self.get_coordinates)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.ContextMenu and source is self.treeWidget:
            active = len(self.treeWidget.selectedItems()) > 0
            active_all = self.treeWidget.topLevelItemCount() > 0
            menu = QtWidgets.QMenu()
            menu.addAction('Plot...', lambda: (self.plot_minima())).setEnabled(active or active_all)
            menu.addAction('Delete', lambda: (self.remove_from_tree())).setEnabled(active)
            menu.addSeparator()
            menu.addAction('Import...', lambda: (self.delete()))
            menu.addAction('Export...', lambda: (self.export_minimas())).setEnabled(active or active_all)
            menu.exec_(event.globalPos())
            return True

        if event.type() == QtCore.QEvent.ContextMenu and source is self.tableWidget:
            active = self.tableWidget.rowCount() > 0
            active_selected = len(self.tableWidget.selectionModel().selectedRows()) > 0
            menu = QtWidgets.QMenu()
            menu.addAction('Delete', lambda: (self.delete_boundaries())).setEnabled(active_selected)
            menu.addSeparator()
            menu.addAction('Import...', lambda: (self.import_boundaries()))
            menu.addAction('Export...', lambda: (self.export_boundaries())).setEnabled(active)
            menu.exec_(event.globalPos())
            return True

        return super(MinimumForm, self).eventFilter(source, event)

    def delete_boundaries(self):
        rows_to_delete = []
        for row in self.tableWidget.selectionModel().selectedRows():
            rows_to_delete.append(row.row())

        for row in reversed(rows_to_delete):
            self.tableWidget.removeRow(row)

    def export_boundaries(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "File to Save", "boundaries.csv",
                                                             "Comma-separated values (*.csv)",
                                                             options=options)
        if file_name:
            with open(file_name, "w") as f2w:
                f2w.write("#start_time,end_time\n")
                for ind in range(self.tableWidget.rowCount()):
                    f2w.write(
                        f"{float(self.tableWidget.item(ind, 0).text())},{float(self.tableWidget.item(ind, 1).text())}\n"
                    )

    def import_boundaries(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "File to load", "",
                                                             "Comma-separated values (*.csv)",
                                                             options=options)
        if file_name:
            if self.tableWidget.rowCount() != 0:
                answer = QtWidgets.QMessageBox.question(
                    self.frame,
                    "What to do", "Do you want to clear the old data?",
                    QtWidgets.QMessageBox.Yes |
                    QtWidgets.QMessageBox.No
                )
                if answer == QtWidgets.QMessageBox.Yes:
                    while self.tableWidget.rowCount() > 0:
                        self.tableWidget.removeRow(0)

            with open(file_name, "r") as f2r:
                for line in f2r:
                    if line.startswith("#"):
                        continue
                    try:
                        x, y = list(map(float, line.split(",")))
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "Error", str(e))
                        return

                    count = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(count)
                    self.tableWidget.setItem(count, 0, QtWidgets.QTableWidgetItem(str(x)))
                    self.tableWidget.setItem(count, 1, QtWidgets.QTableWidgetItem(str(y)))

    def export_minimas(self):
        data = {}
        for i in range(self.treeWidget.topLevelItemCount()):
            each = []
            for u in range(self.treeWidget.topLevelItem(i).childCount()):
                each.append(float(self.treeWidget.topLevelItem(i).child(u).text(0)))
            data[self.treeWidget.topLevelItem(i).text(0)] = each

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "File to Save", "minimas.json",
                                                             "JavaScript Object Notation (*.json)",
                                                             options=options)
        if file_name:
            with open(file_name, "w") as f2w:
                json.dump(data, f2w)

    def remove_from_tree(self):
        for item in self.treeWidget.selectedItems():
            (item.parent() or self.treeWidget.invisibleRootItem()).removeChild(item)

    def plot_minima(self):
        data = []
        for item in self.treeWidget.selectedItems():
            if item.parent() is None:
                data.append(
                    [
                        item.text(0),
                        [
                            float(item.child(i).text(0))
                            for i in range(item.childCount())
                        ]
                    ]
                )

        def f(ax):
            legends = []
            self.xlc.plot(ax=ax)
            color_sample = sample(DARK_COLORS, k=len(data))
            for it, item in enumerate(data):
                c = color_sample[it]
                red_patch = mpatches.Patch(color=c, label=item[0])
                legends.append(red_patch)
                for tm in item[1]:
                    ax.axvline(x=tm, color=c)
            ax.legend(handles=legends)

        graph_to_show = GraphForm(self, f)
        self.parent().window().mdiArea.addSubWindow(graph_to_show)
        graph_to_show.show()

    def get_coordinates(self, event):
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods == QtCore.Qt.AltModifier and event.button == 1:
            self.right = event.xdata
        if mods == QtCore.Qt.ControlModifier and event.button == 1:
            self.left = event.xdata

        if self.left is not None and self.right is not None:
            count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(count)
            self.tableWidget.setItem(count, 0, QtWidgets.QTableWidgetItem(str(self.left)))
            self.tableWidget.setItem(count, 1, QtWidgets.QTableWidgetItem(str(self.right)))
            self.left = None
            self.right = None

        self.highlight_selected()

    def highlight_selected(self):
        indices = set([each.row() for each in self.tableWidget.selectedIndexes()])

        for line in self.sc.axes.lines[1:]:
            line.remove()

        for ind in range(self.tableWidget.rowCount()):
            if ind in indices:
                c1, c2 = "gm"
            else:
                c1, c2 = "by"

            self.sc.axes.axvline(x=float(self.tableWidget.item(ind, 0).text()), color=c1)
            self.sc.axes.axvline(x=float(self.tableWidget.item(ind, 1).text()), color=c2)

        if self.left is not None:
            self.sc.axes.axvline(x=float(self.left), color="b", linestyle=":")

        if self.right is not None:
            self.sc.axes.axvline(x=float(self.right), color="y", linestyle=":")

        self.sc.draw()

    def find_boundaries(self):
        try:
            boundaries = self.xlc.boundaries_extrema(sigma_multiplier=self.doubleSpinBox.value())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(boundaries), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)

        for line in self.sc.axes.lines[1:]:
            line.remove()

        self.tableWidget.clear()

        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)

        for it, boundary in enumerate(boundaries):
            self.sc.axes.axvline(x=boundary[0], color="b")
            self.sc.axes.axvline(x=boundary[1], color="y")
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            self.tableWidget.insertRow(it)
            self.tableWidget.setItem(it, 0, QtWidgets.QTableWidgetItem(str(boundary[0])))
            self.tableWidget.setItem(it, 1, QtWidgets.QTableWidgetItem(str(boundary[1])))
            # QtWidgets.QTreeWidgetItem(self.tableWidget, [str(time), str(flux), str(flux_err)])

        self.sc.draw()
        progress.close()

    def calculate(self):

        try:
            boundaries = [
                [
                    float(self.tableWidget.item(ind, 0).text()),
                    float(self.tableWidget.item(ind, 1).text())
                ]
                for ind in range(self.tableWidget.rowCount())
            ]

            if self.comboBox.currentText() != "Periodogram" and len(boundaries) < 1:
                QtWidgets.QMessageBox.critical(self, "Error", "Boundaries required")
                return

            minimas = {}
            if self.comboBox.currentText() == "Periodogram":
                minimas["periodogram"] = self.xlc.minima("periodogram")
            elif self.comboBox.currentText() == "*Local":
                minimas["local"] = self.xlc.minima("local", boundaries=boundaries)
            elif self.comboBox.currentText() == "*Mean":
                minimas["mean"] = self.xlc.minima("mean", boundaries=boundaries)
            elif self.comboBox.currentText() == "*Median":
                minimas["median"] = self.xlc.minima("median", boundaries=boundaries)
            elif self.comboBox.currentText() == "*Fit":
                minimas["fit"] = self.xlc.minima("fit", boundaries=boundaries)
            elif self.comboBox.currentText() == "*All":
                minimas = self.xlc.minima("all", boundaries=boundaries)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, sum([len(each) for each in minimas.values()]),
                                             self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)

        top_level_items = [
            self.treeWidget.topLevelItem(i).text(0)
            for i in range(self.treeWidget.topLevelItemCount())
        ]

        for kind, values in minimas.items():
            new_kind = kind
            c = 0
            while new_kind in top_level_items:
                c += 1
                new_kind = kind + f"_{c}"

            kind_layer = QtWidgets.QTreeWidgetItem(self.treeWidget, [str(new_kind)])
            kind_layer.setFirstColumnSpanned(True)

            for it, value in enumerate(values):
                progress.setValue(it)
                QtWidgets.QApplication.processEvents()
                QtWidgets.QTreeWidgetItem(kind_layer, [str(value)])

        progress.close()


class PeriodogramFoldForm(QtWidgets.QWidget, fold_periodogram.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(PeriodogramFoldForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):
        if self.comboBox.currentText() == "Parts Per Thousand":
            unit = "ppt"
        else:
            unit = "ppm"

        try:
            self.parent().window().xlc.append(
                self.xlc.fold_periodogram(unit=unit)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class PhaseFoldForm(QtWidgets.QWidget, fold_phase.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(PhaseFoldForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):

        try:
            self.parent().window().xlc.append(
                self.xlc.fold_phase(Time(self.doubleSpinBox.value(), scale="tdb", format="jd"),
                                    self.doubleSpinBox_2.value())
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class NormalizeForm(QtWidgets.QWidget, graph.Ui_Form):
    def __init__(self, parent, xlc):
        super(NormalizeForm, self).__init__(parent)
        self.setupUi(self)

        self.xlc = xlc

        self.sc = MplCanvas(5, 4, 100)
        self.xlc.normalize("ppt").plot(ax=self.sc.axes)
        toolbar = NavigationToolbar(self.sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)

        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.addItem("Parts Per Thousand")
        self.comboBox.addItem("Parts Per Million")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.addWidget(widget, 0, 0)
        self.gridLayout.addWidget(self.comboBox, 1, 0)
        self.comboBox.currentTextChanged.connect(lambda: (self.calculate()))
        self.show()

    def calculate(self):
        try:
            self.sc.axes.clear()
            if self.comboBox.currentText() == "Parts Per Thousand":
                self.xlc.normalize(unit='ppt').plot(ax=self.sc.axes)
            else:
                self.xlc.normalize(unit='ppm').plot(ax=self.sc.axes)

            self.sc.draw()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return


class PeriodogramForm(QtWidgets.QWidget, graph.Ui_Form):
    def __init__(self, parent, xlc):
        super(PeriodogramForm, self).__init__(parent)
        self.setupUi(self)

        self.xlc = xlc

        self.sc = MplCanvas(5, 4, 100)
        self.xlc.normalize("ppt").to_periodogram().plot(ax=self.sc.axes)
        toolbar = NavigationToolbar(self.sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)

        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.addItem("Parts Per Thousand")
        self.comboBox.addItem("Parts Per Million")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.addWidget(widget, 0, 0)
        self.gridLayout.addWidget(self.comboBox, 1, 0)
        self.comboBox.currentTextChanged.connect(lambda: (self.calculate()))
        self.show()

    def calculate(self):
        try:
            self.sc.axes.clear()
            if self.comboBox.currentText() == "Parts Per Thousand":
                self.xlc.normalize(unit='ppt').to_periodogram().plot(ax=self.sc.axes)
            else:
                self.xlc.normalize(unit='ppm').to_periodogram().plot(ax=self.sc.axes)

            self.sc.draw()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return


class SavitzkyGolayForm(QtWidgets.QWidget, savitzky_golay.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(SavitzkyGolayForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):
        window_size = self.spinBox.value()
        order = self.spinBox_2.value()
        deriv = self.spinBox_3.value()
        rate = self.spinBox_4.value()
        try:
            self.parent().window().xlc.append(
                self.xlc.smooth_savitzky_golay(window_size=window_size, order=order, deriv=deriv, rate=rate)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class SavgolFilterForm(QtWidgets.QWidget, savgol_filter.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(SavgolFilterForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):
        window = self.spinBox.value()
        order = self.spinBox_2.value()

        try:
            self.parent().window().xlc.append(
                self.xlc.smooth_savgol_filter(window=window, order=order)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class ButterworthFilterForm(QtWidgets.QWidget, butterworth_filter.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(ButterworthFilterForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):
        cutoff_freq = self.doubleSpinBox.value()
        sampling_rate = self.doubleSpinBox_2.value()
        order = self.spinBox.value()

        try:
            self.parent().window().xlc.append(
                self.xlc.smooth_butterworth_filter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate, order=order)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class BSplineForm(QtWidgets.QWidget, b_spline.Ui_Dialog):
    def __init__(self, parent, xlc):
        super(BSplineForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):
        s = self.doubleSpinBox.value()
        try:
            self.parent().window().xlc.append(
                self.xlc.smooth_b_spline(s=s)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class TimeTravelForm(QtWidgets.QWidget, time_travel.Ui_Dialog):
    def __init__(self, parent, xlc, item, time_travel_type):
        super(TimeTravelForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc
        self.item = item
        self.time_travel_type = time_travel_type

        self.pushButton.clicked.connect(lambda: (self.calculate()))
        self.comboBox.currentTextChanged.connect(lambda: (self.get_location()))
        self.pushButton_2.clicked.connect(lambda: (self.get_coordinates()))
        self.set_all_earth_locations()

        if self.time_travel_type == "hjd":
            self.setWindowTitle("Heliocentric JD")
        elif self.time_travel_type == "bjd":
            self.setWindowTitle("Barycentric JD")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Unknown Time Travel Type")
            return

    def get_coordinates(self):
        try:
            self.lineEdit_2.setText("")
            self.lineEdit_3.setText("")

            obj_name = self.lineEdit.text()
            if obj_name != "":
                sky_obj = SkyCoord.from_name(obj_name)
                self.lineEdit_2.setText(str(sky_obj.ra.deg))
                self.lineEdit_3.setText(str(sky_obj.dec.deg))
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please provide name")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

    def get_location(self):
        try:
            self.lineEdit_4.setText("")
            self.lineEdit_5.setText("")
            self.lineEdit_6.setText("")

            location_name = self.comboBox.currentText()
            location = EarthLocation.of_site(location_name)
            self.lineEdit_4.setText(str(location.lat.deg))
            self.lineEdit_5.setText(str(location.lon.deg))
            self.lineEdit_6.setText(str(location.height.to(units.m).value))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

    def set_all_earth_locations(self):
        self.comboBox.clear()
        sites = EarthLocation.get_site_names()
        self.comboBox.addItems(sites)

    def calculate(self):
        try:
            ra = self.lineEdit_2.text()
            dec = self.lineEdit_3.text()
            sky = SkyCoord(ra=float(ra) * units.deg, dec=float(dec) * units.deg)

            lat = self.lineEdit_4.text()
            lon = self.lineEdit_5.text()
            height = self.lineEdit_6.text()
            location = EarthLocation(lat=float(lat) * units.deg, lon=float(lon) * units.deg,
                                     height=float(height) * units.m)
            if self.time_travel_type == "hjd":
                self.parent().window().xlc.append(
                    self.xlc.to_hjd(sky=sky, loc=location)
                )
            elif self.time_travel_type == "bjd":
                self.parent().window().xlc.append(
                    self.xlc.to_bjd(sky=sky, loc=location)
                )
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Unknown Time Travel Type")
                return

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class AddTimeForm(QtWidgets.QWidget, add_time.Ui_Dialog):
    def __init__(self, parent, xlc, item):
        super(AddTimeForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc
        self.item = item

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):

        try:
            if self.comboBox.currentText() == "Days":
                time = TimeDelta(self.doubleSpinBox.value() * units.d)
            elif self.comboBox.currentText() == "Hours":
                time = TimeDelta(self.doubleSpinBox.value() * units.h)
            elif self.comboBox.currentText() == "Minutes":
                time = TimeDelta(self.doubleSpinBox.value() * units.min)
            elif self.comboBox.currentText() == "Seconds":
                time = TimeDelta(self.doubleSpinBox.value() * units.s)
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Cannot pars time")
                return

            self.xlc.update_time(time)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.xlc), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for i in range(self.item.childCount()):
            progress.setValue(i)
            QtWidgets.QApplication.processEvents()
            self.item.child(i).setText(0, str(self.xlc.time[i].jd))

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class AddFluxForm(QtWidgets.QWidget, add_flux.Ui_Dialog):
    def __init__(self, parent, xlc, item):
        super(AddFluxForm, self).__init__(parent)
        self.setupUi(self)
        self.xlc = xlc
        self.item = item

        self.pushButton.clicked.connect(lambda: (self.calculate()))

    def calculate(self):

        try:
            self.xlc.update_flux(self.doubleSpinBox.value())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.xlc), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for i in range(self.item.childCount()):
            progress.setValue(i)
            QtWidgets.QApplication.processEvents()
            self.item.child(i).setText(1, str(self.xlc.flux[i].value))

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()


class LoadForm(QtWidgets.QWidget, file_loader.Ui_Form):
    def __init__(self, parent):
        super(LoadForm, self).__init__(parent)
        self.setupUi(self)

        self.pushButton.clicked.connect(lambda: (self.open_file()))
        self.lineEdit_2.textChanged.connect(lambda: (self.load_xlightcurve_parameters()))
        self.pushButton_2.clicked.connect(lambda: (self.load_xlightcurve()))

    def load_xlightcurve(self):
        self.pushButton_2.setEnabled(False)

        time_column = self.comboBox.currentText()
        flux_column = self.comboBox_2.currentText()
        flux_error_column = self.comboBox_3.currentText()

        if not time_column or not flux_column:
            QtWidgets.QMessageBox.critical(self, "Error", "Please provide Time and Flux columns")
            self.pushButton_2.setEnabled(True)
            return

        data = pd.read_csv(self.lineEdit.text(), delimiter=self.lineEdit_2.text())
        try:
            self.parent().window().xlc.append(
                XLightCurve(
                    time=data[time_column],
                    flux=data[flux_column],
                    flux_err=data[flux_error_column] if flux_error_column else flux_error_column
                )
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.pushButton_2.setEnabled(True)
            return

        lxc_layer = QtWidgets.QTreeWidgetItem(self.parent().window().treeWidget,
                                              [str(self.parent().window().xlc[-1]), "", ""])
        lxc_layer.setFirstColumnSpanned(True)

        progress = QtWidgets.QProgressDialog('Work in progress', '', 0, len(self.parent().window().xlc[-1]), self,
                                             QtCore.Qt.FramelessWindowHint)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        for it, (time, flux, flux_err) in enumerate(self.parent().window().xlc[-1]):
            progress.setValue(it)
            QtWidgets.QApplication.processEvents()
            QtWidgets.QTreeWidgetItem(lxc_layer, [str(time), str(flux), str(flux_err)])

        progress.close()

        for window in self.parent().window().mdiArea.subWindowList():
            if window.widget() is self:
                window.close()

    def load_xlightcurve_parameters(self):
        self.comboBox.clear()
        self.comboBox_2.clear()
        self.comboBox_3.clear()

        delimeter = self.lineEdit_2.text()

        if delimeter:
            headers = self.plainTextEdit.toPlainText().split("\n")[0].split(delimeter)
            self.comboBox.addItems(headers)
            self.comboBox_2.addItems(headers)
            self.comboBox_3.addItems(headers)

    def open_file(self):
        self.plainTextEdit.clear()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "File to load", "",
                                                             "Comma-separated values (*.csv);;Data File (*.dat)",
                                                             options=options)
        try:
            if file_name:
                with open(file_name, "r") as f2r:
                    head = "".join([next(f2r) for _ in range(5)])
                    self.plainTextEdit.insertPlainText(head)

                self.load_xlightcurve_parameters()
                self.lineEdit.setText(file_name)
        except Exception as e:
            if e == "":
                e = "Error 418: Confused Computer Syndrome."
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

    def closeEvent(self, event):
        self.parent().window().loader_form = None


def main():
    qdarktheme.enable_hi_dpi()
    app = QtWidgets.QApplication(argv)
    qdarktheme.setup_theme("auto")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()