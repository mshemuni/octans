# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\gui\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 700)
        MainWindow.setMinimumSize(QtCore.QSize(900, 700))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName("mdiArea")
        self.gridLayout_2.addWidget(self.mdiArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget_3 = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget_3.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget_3.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidget_3.setObjectName("dockWidget_3")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.gridLayout = QtWidgets.QGridLayout(self.dockWidgetContents_3)
        self.gridLayout.setObjectName("gridLayout")
        self.treeWidget = QtWidgets.QTreeWidget(self.dockWidgetContents_3)
        self.treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.treeWidget.setHeaderHidden(False)
        self.treeWidget.setObjectName("treeWidget")
        self.gridLayout.addWidget(self.treeWidget, 0, 0, 1, 2)
        self.dockWidget_3.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_3)
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionClear = QtWidgets.QAction(MainWindow)
        self.actionClear.setObjectName("actionClear")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionPeriodogram = QtWidgets.QAction(MainWindow)
        self.actionPeriodogram.setObjectName("actionPeriodogram")
        self.actionSavitzky_Golay = QtWidgets.QAction(MainWindow)
        self.actionSavitzky_Golay.setObjectName("actionSavitzky_Golay")
        self.actionB_Spline = QtWidgets.QAction(MainWindow)
        self.actionB_Spline.setObjectName("actionB_Spline")
        self.actionSavgol_Filter = QtWidgets.QAction(MainWindow)
        self.actionSavgol_Filter.setObjectName("actionSavgol_Filter")
        self.actionButterworth_Filter = QtWidgets.QAction(MainWindow)
        self.actionButterworth_Filter.setObjectName("actionButterworth_Filter")
        self.actionExtrema = QtWidgets.QAction(MainWindow)
        self.actionExtrema.setObjectName("actionExtrema")
        self.actionMean = QtWidgets.QAction(MainWindow)
        self.actionMean.setObjectName("actionMean")
        self.actionMedian = QtWidgets.QAction(MainWindow)
        self.actionMedian.setObjectName("actionMedian")
        self.actionLocal = QtWidgets.QAction(MainWindow)
        self.actionLocal.setObjectName("actionLocal")
        self.actionPeriodogram_2 = QtWidgets.QAction(MainWindow)
        self.actionPeriodogram_2.setObjectName("actionPeriodogram_2")
        self.actionHJD = QtWidgets.QAction(MainWindow)
        self.actionHJD.setObjectName("actionHJD")
        self.actionBJD = QtWidgets.QAction(MainWindow)
        self.actionBJD.setObjectName("actionBJD")
        self.actionPeriodogram_3 = QtWidgets.QAction(MainWindow)
        self.actionPeriodogram_3.setObjectName("actionPeriodogram_3")
        self.actionPeriodogram_4 = QtWidgets.QAction(MainWindow)
        self.actionPeriodogram_4.setObjectName("actionPeriodogram_4")
        self.actionEpoch = QtWidgets.QAction(MainWindow)
        self.actionEpoch.setObjectName("actionEpoch")
        self.actionAdd = QtWidgets.QAction(MainWindow)
        self.actionAdd.setObjectName("actionAdd")
        self.actionAdd_2 = QtWidgets.QAction(MainWindow)
        self.actionAdd_2.setObjectName("actionAdd_2")
        self.actionPlot = QtWidgets.QAction(MainWindow)
        self.actionPlot.setObjectName("actionPlot")
        self.actionFit = QtWidgets.QAction(MainWindow)
        self.actionFit.setObjectName("actionFit")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionQuit)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Xtrema3.1"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.treeWidget.headerItem().setText(0, _translate("MainWindow", "Time"))
        self.treeWidget.headerItem().setText(1, _translate("MainWindow", "Flux"))
        self.treeWidget.headerItem().setText(2, _translate("MainWindow", "Flux Error"))
        self.actionLoad.setText(_translate("MainWindow", "Load..."))
        self.actionClear.setText(_translate("MainWindow", "Clear"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionPeriodogram.setText(_translate("MainWindow", "Periodogram..."))
        self.actionSavitzky_Golay.setText(_translate("MainWindow", "Savitzky Golay..."))
        self.actionB_Spline.setText(_translate("MainWindow", "B Spline..."))
        self.actionSavgol_Filter.setText(_translate("MainWindow", "Savgol Filter..."))
        self.actionButterworth_Filter.setText(_translate("MainWindow", "Butterworth Filter..."))
        self.actionExtrema.setText(_translate("MainWindow", "Extrema..."))
        self.actionMean.setText(_translate("MainWindow", "Mean..."))
        self.actionMedian.setText(_translate("MainWindow", "Median..."))
        self.actionLocal.setText(_translate("MainWindow", "Local..."))
        self.actionPeriodogram_2.setText(_translate("MainWindow", "Periodogram..."))
        self.actionHJD.setText(_translate("MainWindow", "HJD..."))
        self.actionBJD.setText(_translate("MainWindow", "BJD..."))
        self.actionPeriodogram_3.setText(_translate("MainWindow", "Periodogram..."))
        self.actionPeriodogram_4.setText(_translate("MainWindow", "Periodogram..."))
        self.actionEpoch.setText(_translate("MainWindow", "Epoch..."))
        self.actionAdd.setText(_translate("MainWindow", "Add..."))
        self.actionAdd_2.setText(_translate("MainWindow", "Add..."))
        self.actionPlot.setText(_translate("MainWindow", "Plot..."))
        self.actionFit.setText(_translate("MainWindow", "Fit"))
        self.actionAbout.setText(_translate("MainWindow", "About"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())