"""
VMD rPPG Signal Processor - GUI Implementation
PySide6 desktop application for VMD-based signal processing
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QListWidget, QTabWidget, QMessageBox, QProgressDialog,
    QComboBox, QCheckBox, QGroupBox, QGridLayout, QSplitter, QScrollArea, QSizePolicy,
    QToolButton, QFrame, QProgressBar, QDialog, QDialogButtonBox, QFormLayout, QStackedWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QParallelAnimationGroup, QTimer, QSize
from PySide6.QtGui import QFont, QPixmap, QImage
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.signal import welch, spectrogram
from PIL import Image

import config
import computation
import filters


def run_vmd_computation(params):
    """Worker function for VMD computation"""
    signal = params['signal']
    fps = params['fps']
    original_signal = signal.copy()
    preprocessing_info = []
    
    # Detrend
    if params['detrend']['enabled']:
        signal = computation.detrend_signal(signal, params['detrend']['lambda'])
        preprocessing_info.append(f"Detrend (λ={params['detrend']['lambda']})")
        
    # Normalize (Always enabled now)
    method = 'z-score'
    signal = computation.normalize_signal(signal, method)
    # No longer adding this to the label to keep it clean
    # preprocessing_info.append(f"Normalize ({method})")
        
    # Harmonics
    if params['harmonics']['enabled']:
        gain = params['harmonics']['gain']
        freq_min = params['harmonics']['freq_min']
        freq_max = params['harmonics']['freq_max']
        signal = computation.enhance_harmonics(signal, fps, freq_min, freq_max, gain)
        preprocessing_info.append(f"Harmonics (Gain={gain})")
        
    # VMD
    vmd_p = params['vmd_params']
    modes, center_freqs = computation.run_vmd(
        signal, vmd_p['k'], vmd_p['alpha'], vmd_p['tau'], 
        vmd_p['dc'], vmd_p['init'], vmd_p['tol'], fps
    )
    
    # Selection
    sel_p = params['selection_params']
    selected, mode_info = computation.select_modes(modes, center_freqs, signal, sel_p, fps)
    
    reconstructed = np.sum(modes, axis=0)
    extracted = np.sum(modes[selected], axis=0) if np.any(selected) else np.zeros_like(signal)
    
    return {
        'original': original_signal, 
        'preprocessed': signal,
        'preprocessing_label': " + ".join(preprocessing_info) if preprocessing_info else "None",
        'modes': modes, 
        'mode_info': mode_info, 
        'selected': selected,
        'reconstructed': reconstructed, 
        'extracted': extracted,
        'hr': computation.estimate_heart_rate(extracted, fps),
        'snr': computation.calculate_snr(original_signal, extracted, fps)
    }


def run_filters_computation(params):
    """Worker function for Filters computation"""
    signal = params['signal']
    fps = params['fps']
    original_signal = signal.copy()

    # --- Create preprocessed signal for comparison ---
    preprocessed_signal = original_signal.copy()
    preproc_label_parts = []
    if params['detrend']['enabled']:
        preprocessed_signal = computation.detrend_signal(preprocessed_signal, params['detrend']['lambda'])
        preproc_label_parts.append("Detrend")
    
    # Normalize (Always enabled)
    preprocessed_signal = computation.normalize_signal(preprocessed_signal, 'z-score')
    preproc_label_parts.append("Norm")
        
    if params['harmonics']['enabled']:
        gain = params['harmonics']['gain']
        freq_min = params['harmonics']['freq_min']
        freq_max = params['harmonics']['freq_max']
        preprocessed_signal = computation.enhance_harmonics(preprocessed_signal, fps, freq_min, freq_max, gain)
        preproc_label_parts.append("Harmonics")
        
    preproc_label = " + ".join(preproc_label_parts) if preproc_label_parts else "None"
    
    results = {'preprocessed_signal': preprocessed_signal, 'preproc_label': preproc_label}
    
    # --- Apply all other filters to the ORIGINAL raw signal ---

    # Filter Harmonics (Specific to filter tab)
    if params['filter_harmonics']['enabled']:
        gain = params['filter_harmonics']['gain']
        freq_min = params['filter_harmonics']['freq_min']
        freq_max = params['filter_harmonics']['freq_max']
        enhanced_sig = computation.enhance_harmonics(original_signal, fps, freq_min, freq_max, gain)
        results['Harmonics Enhanced'] = {
            'signal': enhanced_sig, 
            'hr': computation.estimate_heart_rate(enhanced_sig, fps), 
            'snr': computation.calculate_snr(original_signal, enhanced_sig, fps)
        }

    # Butterworth
    if params['butterworth']['enabled']:
        p = params['butterworth']
        filtered = filters.apply_butterworth(original_signal, p['order'], p['freq_min'], p['freq_max'], fps)
        results['Butterworth'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }
        
    # Chebyshev
    if params['chebyshev']['enabled']:
        p = params['chebyshev']
        filtered = filters.apply_chebyshev(original_signal, p['order'], p['ripple'], p['freq_min'], p['freq_max'], fps)
        results['Chebyshev'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }
        
    # Chebyshev II
    if params['cheby2']['enabled']:
        p = params['cheby2']
        filtered = filters.apply_cheby2(original_signal, p['order'], p['stopband_atten'], p['freq_min'], p['freq_max'], fps)
        results['Chebyshev II'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }

    # Elliptic
    if params['elliptic']['enabled']:
        p = params['elliptic']
        filtered = filters.apply_elliptic(original_signal, p['order'], p['passband_ripple'], p['stopband_atten'], p['freq_min'], p['freq_max'], fps)
        results['Elliptic'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }
        
    # Moving Average
    if params['moving_average']['enabled']:
        filtered = filters.apply_moving_average(original_signal, params['moving_average']['window_size'])
        results['Moving Average'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }
        
    # Savgol
    if params['savgol']['enabled']:
        p = params['savgol']
        filtered = filters.apply_savgol(original_signal, p['window_size'], p['poly_order'])
        results['Savitzky-Golay'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }
        
    # Wavelet
    if params['wavelet']['enabled']:
        p = params['wavelet']
        filtered = filters.apply_wavelet(original_signal, p['wavelet'], p['level'], p['threshold_mode'])
        results['Wavelet Denoising'] = {
            'signal': filtered, 
            'hr': computation.estimate_heart_rate(filtered, fps), 
            'snr': computation.calculate_snr(original_signal, filtered, fps)
        }
        
    return results


class ComputationThread(QThread):
    """Generic worker thread for running computations"""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, target, params):
        super().__init__()
        self.target = target
        self.params = params

    def run(self):
        try:
            result = self.target(self.params)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class OptimizationThread(QThread):
    """Thread for running VMD optimization without freezing UI"""
    progress = Signal(int, int)
    finished = Signal(int, float, float)

    def __init__(self, sig, K_range, alpha_range, metric, selection_params,
                 fps, tau, DC, init, tol, fft_size):
        super().__init__()
        self.sig = sig
        self.K_range = K_range
        self.alpha_range = alpha_range
        self.metric = metric
        self.selection_params = selection_params
        self.fps = fps
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.fft_size = fft_size

    def run(self):
        def progress_callback(current, total):
            self.progress.emit(current, total)

        best_K, best_alpha, best_score = computation.auto_optimize_vmd(
            self.sig, self.K_range, self.alpha_range, self.metric,
            self.selection_params, self.fps, self.tau, self.DC,
            self.init, self.tol, self.fft_size, progress_callback
        )

        self.finished.emit(best_K, best_alpha, best_score)


class DynamicTabWidget(QTabWidget):
    """
    A QTabWidget that resizes its height based on the currently active tab's content,
    ignoring the height of other tabs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self.updateGeometry)

    def sizeHint(self):
        current = self.currentWidget()
        if not current:
            return super().sizeHint()
        
        size = current.sizeHint()
        overhead = self.tabBar().sizeHint().height() + 10 
        return QSize(super().sizeHint().width(), size.height() + overhead)

    def minimumSizeHint(self):
        return QSize(0, 0)


class CollapsibleBox(QWidget):
    """
    A simple, robust collapsible widget.
    """
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; font-size: 11pt; background-color: #f0f0f0; padding: 4px; text-align: left; }"
            "QToolButton:hover { background-color: #e0e0e0; }"
            "QToolButton:pressed { background-color: #d0d0d0; }"
        )
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.toggled.connect(self.on_toggled)
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

    def on_toggled(self, checked):
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content_area.setVisible(checked)
        QTimer.singleShot(10, self.update_layout)

    def update_layout(self):
        parent = self.parentWidget()
        while parent:
            parent.updateGeometry()
            if parent.layout():
                parent.layout().activate()
            
            if isinstance(parent, QTabWidget):
                parent.updateGeometry()
                parent.adjustSize()

            if isinstance(parent, (QMainWindow, QScrollArea)):
                break
            parent = parent.parentWidget()
        
        QApplication.processEvents()

    def setContentLayout(self, layout):
        if self.content_area.layout():
            QWidget().setLayout(self.content_area.layout()) 
        self.content_area.setLayout(layout)


class RowCanvas(FigureCanvasQTAgg):
    """Optimized Matplotlib canvas containing 3 subplots for a single row"""
    def __init__(self, title="", parent=None, width=1, height=3):
        fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title = title # Store title for saving figures
        
        # Create 3 subplots with specific ratios
        gs = fig.add_gridspec(1, 3, width_ratios=[6, 2, 2])
        self.ax_sig = fig.add_subplot(gs[0])
        self.ax_spec = fig.add_subplot(gs[1])
        self.ax_psd = fig.add_subplot(gs[2])
        
        # Adjust spacing - tighter defaults
        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.10, wspace=0.15)


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Legacy Matplotlib canvas for embedding plots (kept for compatibility if needed)"""
    def __init__(self, parent=None, width=8, height=3):
        fig = Figure(figsize=(width, height), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumHeight(250) 
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class SaveDataDialog(QDialog):
    def __init__(self, parent=None, vmd_available=False, filters_available=False, initial_save_path=""):
        super().__init__(parent)
        self.setWindowTitle("Save Data")
        
        layout = QVBoxLayout(self)
        
        # Path selection
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a base directory...")
        self.path_edit.setText(initial_save_path) # Set initial path
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        
        # New folder
        form_layout = QFormLayout()
        self.folder_name_edit = QLineEdit()
        self.folder_name_edit.setPlaceholderText("e.g., 'run_01_subject_A' (optional)")
        form_layout.addRow("New Folder Name:", self.folder_name_edit)
        layout.addLayout(form_layout)
        
        # Checkboxes
        self.options_group = QGroupBox("Data to Save")
        options_layout = QVBoxLayout()
        
        self.save_settings_check = QCheckBox("Settings (.json)")
        self.save_settings_check.setChecked(True)
        
        self.save_figure_check = QCheckBox("Save Plots as Image (.png)")
        self.save_figure_check.setChecked(True)
        
        self.save_vmd_check = QCheckBox("VMD Analysis Signals (.csv)")
        self.save_vmd_check.setEnabled(vmd_available)
        
        self.save_imfs_check = QCheckBox("Individual VMD Modes (IMFs) (.csv)")
        self.save_imfs_check.setEnabled(vmd_available)
        
        self.save_filters_check = QCheckBox("Filter Comparison Signals (.csv)")
        self.save_filters_check.setEnabled(filters_available)
        
        self.save_pipeline_check = QCheckBox("Filter Combination (Pipeline) Signals (.csv)")
        self.save_pipeline_check.setEnabled(parent.pipeline_chart_rows is not None and len(parent.pipeline_chart_rows) > 0)
        
        options_layout.addWidget(self.save_settings_check)
        options_layout.addWidget(self.save_figure_check)
        options_layout.addWidget(self.save_vmd_check)
        options_layout.addWidget(self.save_imfs_check)
        options_layout.addWidget(self.save_filters_check)
        options_layout.addWidget(self.save_pipeline_check)
        
        self.options_group.setLayout(options_layout)
        layout.addWidget(self.options_group)
        
        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def browse(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.path_edit.setText(directory)
            
    def get_save_options(self):
        return {
            "base_path": self.path_edit.text(),
            "folder_name": self.folder_name_edit.text(),
            "save_settings": self.save_settings_check.isChecked(),
            "save_figure": self.save_figure_check.isChecked(),
            "save_vmd": self.save_vmd_check.isChecked(),
            "save_imfs": self.save_imfs_check.isChecked(),
            "save_filters": self.save_filters_check.isChecked(),
            "save_pipeline": self.save_pipeline_check.isChecked()
        }


class VMDrPPGMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = config.load_config()
        if 'ui_state' not in self.config:
            self.config['ui_state'] = {}
        self.selected_file_path = None
        self.signal_data = None
        self.vmd_results = None
        self.filter_results = None
        self.current_directory = "./data" if os.path.exists("./data") else "."
        self.shared_x_axis = None
        self.shared_y_axis = None
        self.shared_psd_axis = None
        
        # List to keep track of shared signal control widgets for synchronization
        self.shared_controls = []

        # Separate lists for chart rows for each tab
        self.vmd_chart_rows = []
        self.filter_chart_rows = []
        self.pipeline_chart_rows = []
        self.pipeline_steps = []
        
        self.init_ui()
        self.update_ui_from_config()

    def init_ui(self):
        self.setWindowTitle("VMD and Filters Visualizer") # Changed application title
        self.setGeometry(100, 100, 1400, 800)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget) 
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.left_panel = self.create_left_panel()
        self.splitter.addWidget(self.left_panel)
        right_panel = self.create_right_panel()
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([250, 1150])
        main_layout.addWidget(self.splitter)

    def toggle_sidebar(self):
        self.left_panel.setVisible(not self.left_panel.isVisible())

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.browse_btn = QPushButton("📁 Browse Path")
        self.browse_btn.clicked.connect(self.browse_directory)
        layout.addWidget(self.browse_btn)
        
        self.path_label = QLabel("No directory selected")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.path_label)
        file_list_label = QLabel("CSV Files:")
        file_list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(file_list_label)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        # Task 4: Highlight selected subject in blue
        self.file_list.setStyleSheet("QListWidget::item:selected { background-color: #2563eb; color: white; }")
        layout.addWidget(self.file_list)
        if os.path.exists(self.current_directory):
            self.load_csv_files(self.current_directory)
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(6) 

        # Tabs are now the top-level container for configuration
        self.tabs = DynamicTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        vmd_tab = self.create_vmd_tab()
        self.tabs.addTab(vmd_tab, "VMD Analysis")

        comparison_tab = self.create_comparison_tab()
        self.tabs.addTab(comparison_tab, "Filter Comparison")

        combination_tab = self.create_combination_tab()
        self.tabs.addTab(combination_tab, "Filter Combination")

        layout.addWidget(self.tabs, 0) 

        # Create a stacked widget to hold the results for each tab
        self.results_stack = QStackedWidget()
        
        # VMD Results Area
        self.vmd_results_area = QScrollArea()
        self.vmd_results_area.setWidgetResizable(True)
        self.vmd_results_area.setFrameShape(QFrame.NoFrame)
        self.vmd_results_widget = QWidget()
        self.vmd_results_layout = QVBoxLayout(self.vmd_results_widget)
        self.vmd_results_layout.setSpacing(0)
        self.vmd_results_layout.setContentsMargins(0, 0, 0, 0)
        self.vmd_results_area.setWidget(self.vmd_results_widget)
        self.results_stack.addWidget(self.vmd_results_area)

        # Filter Results Area
        self.filter_results_area = QScrollArea()
        self.filter_results_area.setWidgetResizable(True)
        self.filter_results_area.setFrameShape(QFrame.NoFrame)
        self.filter_results_widget = QWidget()
        self.filter_results_layout = QVBoxLayout(self.filter_results_widget)
        self.filter_results_layout.setSpacing(0)
        self.filter_results_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_results_area.setWidget(self.filter_results_widget)
        self.results_stack.addWidget(self.filter_results_area)

        # Pipeline (Combination) Results Area
        self.pipeline_results_area = QScrollArea()
        self.pipeline_results_area.setWidgetResizable(True)
        self.pipeline_results_area.setFrameShape(QFrame.NoFrame)
        self.pipeline_results_widget = QWidget()
        self.pipeline_results_layout = QVBoxLayout(self.pipeline_results_widget)
        self.pipeline_results_layout.setSpacing(0)
        self.pipeline_results_layout.setContentsMargins(0, 0, 0, 0)
        self.pipeline_results_area.setWidget(self.pipeline_results_widget)
        self.results_stack.addWidget(self.pipeline_results_area)
        
        layout.addWidget(self.results_stack, 1)

        self.tabs.currentChanged.connect(self.results_stack.setCurrentIndex)

        return panel

    def on_tab_changed(self, index):
        # This function is now handled by connecting the tab's currentChanged signal
        # directly to the QStackedWidget's setCurrentIndex slot.
        # We can add any additional logic here if needed when a tab changes.
        pass

    def clear_vmd_results(self):
        self.shared_x_axis = None # Reset shared axes when clearing
        self.shared_psd_axis = None
        if not hasattr(self, 'vmd_results_layout') or self.vmd_results_layout is None:
            return
        while self.vmd_results_layout.count():
            child = self.vmd_results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.vmd_chart_rows = [] # Clear the list of chart rows

    def clear_filter_results(self):
        self.shared_x_axis = None # Reset shared axes when clearing
        self.shared_psd_axis = None
        if not hasattr(self, 'filter_results_layout') or self.filter_results_layout is None:
            return
        while self.filter_results_layout.count():
            child = self.filter_results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.filter_chart_rows = [] # Clear the list of chart rows

    def clear_pipeline_results(self):
        self.shared_x_axis = None
        self.shared_psd_axis = None
        if not hasattr(self, 'pipeline_results_layout') or self.pipeline_results_layout is None:
            return
        while self.pipeline_results_layout.count():
            child = self.pipeline_results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.pipeline_chart_rows = []

    def create_combination_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Config Box
        config_box = CollapsibleBox("Pipeline Configuration")
        config_content = QWidget()
        config_layout = QVBoxLayout(config_content)
        
        # Subject Info (Shared)
        config_layout.addWidget(self.create_shared_info_widget())
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        config_layout.addWidget(line)

        # Controls to add filters
        add_controls = QHBoxLayout()
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems([
            "High Pass", "Low Pass", "Notch",
            "Butterworth (BP)", "Chebyshev I", "Chebyshev II", "Elliptic", 
            "Moving Average", "Savitzky-Golay", "Wavelet", "Harmonics Enhancement"
        ])
        add_btn = QPushButton("➕ Add Filter to Pipeline")
        add_btn.clicked.connect(self.add_pipeline_step)
        add_btn.setStyleSheet("QPushButton { background-color: #3b82f6; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }")
        
        self.run_pipeline_btn = QPushButton("▶ Run Pipeline")
        self.run_pipeline_btn.clicked.connect(self.run_pipeline)
        self.run_pipeline_btn.setStyleSheet("QPushButton { background-color: #10b981; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; }")

        add_controls.addWidget(QLabel("Select Filter:"))
        add_controls.addWidget(self.filter_type_combo)
        add_controls.addWidget(add_btn)
        add_controls.addStretch()
        add_controls.addWidget(self.run_pipeline_btn)
        
        config_layout.addLayout(add_controls)
        
        # Container for pipeline steps
        self.pipeline_steps_container = QWidget()
        self.pipeline_steps_layout = QVBoxLayout(self.pipeline_steps_container)
        self.pipeline_steps_layout.setContentsMargins(0, 5, 0, 5)
        self.pipeline_steps_layout.setSpacing(5)
        self.pipeline_steps_layout.addStretch() # Initial stretch
        
        config_layout.addWidget(self.pipeline_steps_container)
        
        config_box.setContentLayout(config_layout)
        main_layout.addWidget(config_box)
        main_layout.addStretch()

        return tab

    def add_pipeline_step(self):
        filter_type = self.filter_type_combo.currentText()
        step_index = len(self.pipeline_steps)
        
        # Create Step Widget
        step_widget = QFrame()
        step_widget.setFrameShape(QFrame.StyledPanel)
        step_widget.setMinimumHeight(45)
        step_widget.setMaximumHeight(70)
        step_widget.setStyleSheet("""
            QFrame { background-color: #ffffff; border-radius: 4px; border: 1px solid #d1d5db; } 
            QFrame:hover { border: 1px solid #3b82f6; }
        """)
        step_layout = QHBoxLayout(step_widget)
        step_layout.setContentsMargins(5, 1, 5, 1)
        step_layout.setSpacing(8)
        
        # 1. Remove Button (at the front)
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(24, 24)
        del_btn.setToolTip("Remove step")
        del_btn.setStyleSheet("QPushButton { background-color: #fee2e2; color: #ef4444; border: 1px solid #fecaca; border-radius: 12px; font-weight: bold; } QPushButton:hover { background-color: #ef4444; color: white; }")
        del_btn.clicked.connect(lambda: self.remove_pipeline_step(step_widget))
        step_layout.addWidget(del_btn)
        
        # 2. Step label
        lbl = QLabel(f"<b>Step {step_index + 1}:</b>\n{filter_type}")
        lbl.setFixedWidth(120)
        lbl.setStyleSheet("border: none; color: #374151;")
        step_layout.addWidget(lbl)
        
        # 3. Parameters widget (reusing logic but creating fresh widgets)
        params_widget, param_inputs = self.create_step_params_ui(filter_type)
        params_widget.setStyleSheet("border: none; background: transparent;")
        step_layout.addWidget(params_widget, 1) # Give it stretch
        
        # 4. Reorder buttons
        reorder_lay = QVBoxLayout()
        reorder_lay.setSpacing(2)
        move_up = QPushButton("▲")
        move_up.setFixedSize(22, 18)
        move_up.setStyleSheet("font-size: 8px; padding: 0;")
        move_up.clicked.connect(lambda: self.move_pipeline_step(step_widget, -1))
        
        move_down = QPushButton("▼")
        move_down.setFixedSize(22, 18)
        move_down.setStyleSheet("font-size: 8px; padding: 0;")
        move_down.clicked.connect(lambda: self.move_pipeline_step(step_widget, 1))
        
        reorder_lay.addWidget(move_up)
        reorder_lay.addWidget(move_down)
        step_layout.addLayout(reorder_lay)
        
        # Add to UI
        # Insert before the stretch
        self.pipeline_steps_layout.insertWidget(self.pipeline_steps_layout.count() - 1, step_widget)
        
        # Add to state
        self.pipeline_steps.append({
            'type': filter_type,
            'widget': step_widget,
            'params': param_inputs
        })
        
        # Force tab to update height
        self.tabs.updateGeometry()

    def remove_pipeline_step(self, widget):
        for i, step in enumerate(self.pipeline_steps):
            if step['widget'] == widget:
                self.pipeline_steps.pop(i)
                break
        widget.deleteLater()
        QTimer.singleShot(50, self.update_step_numbers)

    def move_pipeline_step(self, widget, delta):
        idx = -1
        for i, step in enumerate(self.pipeline_steps):
            if step['widget'] == widget:
                idx = i
                break
        
        new_idx = idx + delta
        if 0 <= new_idx < len(self.pipeline_steps):
            # Swap in list
            self.pipeline_steps[idx], self.pipeline_steps[new_idx] = self.pipeline_steps[new_idx], self.pipeline_steps[idx]
            
            # Update UI layout
            # Remove stretch first
            stretch = self.pipeline_steps_layout.takeAt(self.pipeline_steps_layout.count() - 1)
            
            # Rebuild layout
            for step in self.pipeline_steps:
                self.pipeline_steps_layout.removeWidget(step['widget'])
            
            for step in self.pipeline_steps:
                self.pipeline_steps_layout.addWidget(step['widget'])
            
            self.pipeline_steps_layout.addItem(stretch)
            self.update_step_numbers()

    def update_step_numbers(self):
        for i, step in enumerate(self.pipeline_steps):
            label = step['widget'].findChild(QLabel)
            if label:
                label.setText(f"<b>Step {i + 1}:</b>\n{step['type']}")

    def create_numeric_control(self, is_double=False, min_val=0, max_val=100, default=0, step=1, decimals=1):
        """Creates a compact control with [ - ] [ Value ] [ + ] buttons"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        spin = QDoubleSpinBox() if is_double else QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        if is_double:
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
        else:
            spin.setSingleStep(int(step))
        
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        spin.setAlignment(Qt.AlignCenter)
        spin.setFixedWidth(45)
        spin.setStyleSheet("border: 1px solid #d1d5db; border-radius: 0px; background: white; height: 18px; font-size: 9pt;")

        btn_minus = QPushButton("-")
        btn_minus.setFixedSize(18, 18)
        btn_minus.setStyleSheet("QPushButton { background: #f3f4f6; border: 1px solid #d1d5db; border-right: none; border-radius: 3px 0 0 3px; font-weight: bold; } QPushButton:hover { background: #e5e7eb; }")
        btn_minus.clicked.connect(lambda: spin.setValue(spin.value() - spin.singleStep()))

        btn_plus = QPushButton("+")
        btn_plus.setFixedSize(18, 18)
        btn_plus.setStyleSheet("QPushButton { background: #f3f4f6; border: 1px solid #d1d5db; border-left: none; border-radius: 0 3px 3px 0; font-weight: bold; } QPushButton:hover { background: #e5e7eb; }")
        btn_plus.clicked.connect(lambda: spin.setValue(spin.value() + spin.singleStep()))

        layout.addWidget(btn_minus)
        layout.addWidget(spin)
        layout.addWidget(btn_plus)
        
        return container, spin

    def create_step_params_ui(self, filter_type):
        """Creates a small parameter UI for a pipeline step"""
        container = QWidget()
        lay = QGridLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        inputs = {}

        def add_input(name, row, col, **kwargs):
            lay.addWidget(QLabel(name), row, col)
            ctrl, spin = self.create_numeric_control(**kwargs)
            lay.addWidget(ctrl, row, col + 1)
            return spin

        if filter_type == "Butterworth (BP)":
            inputs['order'] = add_input("Ord:", 0, 0, min_val=1, max_val=10, default=2)
            inputs['freq_min'] = add_input("FMin:", 0, 2, is_double=True, default=0.7, step=0.1)
            inputs['freq_max'] = add_input("FMax:", 0, 4, is_double=True, default=4.0, step=0.1)

        elif filter_type == "High Pass":
            inputs['order'] = add_input("Ord:", 0, 0, min_val=1, max_val=10, default=2)
            inputs['cutoff'] = add_input("Cut:", 0, 2, is_double=True, default=0.7, step=0.1)

        elif filter_type == "Low Pass":
            inputs['order'] = add_input("Ord:", 0, 0, min_val=1, max_val=10, default=2)
            inputs['cutoff'] = add_input("Cut:", 0, 2, is_double=True, default=4.0, step=0.1)

        elif filter_type == "Notch":
            inputs['cutoff'] = add_input("Freq:", 0, 0, is_double=True, min_val=1, max_val=500, default=50.0, step=1)
            inputs['q'] = add_input("Q:", 0, 2, is_double=True, min_val=1, max_val=100, default=30.0, step=1)

        elif filter_type == "Chebyshev I":
            inputs['order'] = add_input("Ord:", 0, 0, min_val=1, max_val=10, default=2)
            inputs['ripple'] = add_input("Rip:", 0, 2, is_double=True, default=1.0, step=0.1)
            inputs['freq_min'] = add_input("FMin:", 1, 0, is_double=True, default=0.7, step=0.1)
            inputs['freq_max'] = add_input("FMax:", 1, 2, is_double=True, default=4.0, step=0.1)

        elif filter_type == "Chebyshev II":
            inputs['order'] = add_input("Ord:", 0, 0, min_val=1, max_val=10, default=2)
            inputs['stopband_atten'] = add_input("Att:", 0, 2, min_val=20, max_val=100, default=40)
            inputs['freq_min'] = add_input("FMin:", 1, 0, is_double=True, default=0.7, step=0.1)
            inputs['freq_max'] = add_input("FMax:", 1, 2, is_double=True, default=4.0, step=0.1)

        elif filter_type == "Elliptic":
            inputs['order'] = add_input("Ord:", 0, 0, min_val=1, max_val=10, default=2)
            inputs['passband_ripple'] = add_input("PBRip:", 0, 2, is_double=True, default=1.0, step=0.1)
            inputs['stopband_atten'] = add_input("SBAtt:", 0, 4, min_val=20, max_val=100, default=40)
            inputs['freq_min'] = add_input("FMin:", 1, 0, is_double=True, default=0.7, step=0.1)
            inputs['freq_max'] = add_input("FMax:", 1, 2, is_double=True, default=4.0, step=0.1)

        elif filter_type == "Moving Average":
            inputs['window_size'] = add_input("Win:", 0, 0, min_val=3, max_val=101, default=5, step=2)

        elif filter_type == "Savitzky-Golay":
            inputs['window_size'] = add_input("Win:", 0, 0, min_val=5, max_val=101, default=11, step=2)
            inputs['poly_order'] = add_input("Poly:", 0, 2, min_val=1, max_val=10, default=3)

        elif filter_type == "Wavelet":
            lay.addWidget(QLabel("Wav:"), 0, 0)
            inputs['wavelet'] = QComboBox(); inputs['wavelet'].addItems(['db4', 'sym4', 'coif4', 'haar']); inputs['wavelet'].setCurrentText('db4')
            inputs['wavelet'].setFixedWidth(70)
            lay.addWidget(inputs['wavelet'], 0, 1)
            inputs['level'] = add_input("Lev:", 0, 2, min_val=1, max_val=10, default=3)

        elif filter_type == "Harmonics Enhancement":
            inputs['gain'] = add_input("Gain:", 0, 0, is_double=True, min_val=1, max_val=10, default=2.0, step=0.1)
            inputs['freq_min'] = add_input("FMin:", 0, 2, is_double=True, default=0.7, step=0.1)
            inputs['freq_max'] = add_input("FMax:", 0, 4, is_double=True, default=4.0, step=0.1)

        return container, inputs

        return container, inputs

    def run_pipeline(self):
        raw_signal = self.get_selected_signal_segment()
        if raw_signal is None or len(raw_signal) == 0:
            QMessageBox.warning(self, "No Signal Selected", "Please select a CSV file first.")
            return

        self.clear_pipeline_results()
        fps = self.config['fps']
        
        # Step 0: Raw Signal (Zero-Mean)
        current_signal = raw_signal - np.mean(raw_signal)
        hr_raw = computation.estimate_heart_rate(raw_signal, fps)
        snr_raw = computation.calculate_snr(raw_signal, raw_signal, fps)
        
        self.add_chart_row("Initial Raw Signal", current_signal, "#64748b", raw_signal, hr_raw, snr_raw,
                           target_layout=self.pipeline_results_layout,
                           target_chart_rows_list=self.pipeline_chart_rows)

        # Apply each step sequentially
        for i, step in enumerate(self.pipeline_steps):
            f_type = step['type']
            p = step['params']
            
            try:
                if f_type == "Butterworth (BP)":
                    current_signal = filters.apply_butterworth(current_signal, p['order'].value(), p['freq_min'].value(), p['freq_max'].value(), fps)
                elif f_type == "High Pass":
                    current_signal = filters.apply_highpass(current_signal, p['order'].value(), p['cutoff'].value(), fps)
                elif f_type == "Low Pass":
                    current_signal = filters.apply_lowpass(current_signal, p['order'].value(), p['cutoff'].value(), fps)
                elif f_type == "Notch":
                    current_signal = filters.apply_notch(current_signal, p['cutoff'].value(), p['q'].value(), fps)
                elif f_type == "Chebyshev I":
                    current_signal = filters.apply_chebyshev(current_signal, p['order'].value(), p['ripple'].value(), p['freq_min'].value(), p['freq_max'].value(), fps)
                elif f_type == "Chebyshev II":
                    current_signal = filters.apply_cheby2(current_signal, p['order'].value(), p['stopband_atten'].value(), p['freq_min'].value(), p['freq_max'].value(), fps)
                elif f_type == "Elliptic":
                    current_signal = filters.apply_elliptic(current_signal, p['order'].value(), p['passband_ripple'].value(), p['stopband_atten'].value(), p['freq_min'].value(), p['freq_max'].value(), fps)
                elif f_type == "Moving Average":
                    current_signal = filters.apply_moving_average(current_signal, p['window_size'].value())
                elif f_type == "Savitzky-Golay":
                    current_signal = filters.apply_savgol(current_signal, p['window_size'].value(), p['poly_order'].value())
                elif f_type == "Wavelet":
                    current_signal = filters.apply_wavelet(current_signal, p['wavelet'].currentText(), p['level'].value(), 'soft')
                elif f_type == "Harmonics Enhancement":
                    current_signal = computation.enhance_harmonics(current_signal, fps, p['freq_min'].value(), p['freq_max'].value(), p['gain'].value())

                hr = computation.estimate_heart_rate(current_signal, fps)
                snr = computation.calculate_snr(raw_signal, current_signal, fps)
                
                self.add_chart_row(f"Step {i+1}: {f_type}", current_signal, "#2563eb", raw_signal, hr, snr,
                                   target_layout=self.pipeline_results_layout,
                                   target_chart_rows_list=self.pipeline_chart_rows)
                                   
            except Exception as e:
                QMessageBox.critical(self, "Pipeline Error", f"Error at step {i+1} ({f_type}):\n{str(e)}")
                break

        self.finalize_chart_layout(self.pipeline_chart_rows)
        # Switch to the pipeline results area automatically
        self.results_stack.setCurrentIndex(2)

    def create_shared_info_widget(self):
        """Creates a subject info widget and registers its controls for sync"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 4px; border: 1px solid #e5e7eb; }")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)
        
        layout.addWidget(QLabel("<b>File:</b>"))
        file_name_label = QLabel("No file selected")
        file_name_label.setStyleSheet("color: #2563eb; font-weight: bold;")
        file_name_label.setMaximumWidth(250)
        layout.addWidget(file_name_label)
        
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        layout.addWidget(QLabel("<b>Start:</b>"))
        start_time_input = QDoubleSpinBox()
        start_time_input.setRange(0, 10000)
        start_time_input.setValue(self.config.get('signal', {}).get('start_time', 1.0))
        start_time_input.setSingleStep(0.1)
        start_time_input.setFixedWidth(60)
        start_time_input.valueChanged.connect(self.sync_shared_controls)
        layout.addWidget(start_time_input)
        
        layout.addWidget(QLabel("<b>Dur:</b>"))
        duration_input = QDoubleSpinBox()
        duration_input.setRange(0.1, 10000)
        duration_input.setValue(self.config.get('signal', {}).get('duration', 10.0))
        duration_input.setSingleStep(0.1)
        duration_input.setFixedWidth(60)
        duration_input.valueChanged.connect(self.sync_shared_controls)
        layout.addWidget(duration_input)
        
        layout.addWidget(QLabel("<b>FPS:</b>"))
        fps_input = QSpinBox()
        fps_input.setRange(1, 1000)
        fps_input.setValue(self.config['fps'])
        fps_input.setFixedWidth(60)
        fps_input.valueChanged.connect(self.sync_shared_controls)
        layout.addWidget(fps_input)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)
        
        total_rows_label = QLabel("Rows: -")
        layout.addWidget(total_rows_label)
        
        signal_length_label = QLabel("Len: -")
        layout.addWidget(signal_length_label)
        
        # Chart Height Control moved here
        line3 = QFrame()
        line3.setFrameShape(QFrame.VLine)
        line3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line3)
        
        layout.addWidget(QLabel("<b>Graph Height:</b>"))
        chart_height_spin = QSpinBox()
        chart_height_spin.setRange(100, 800)
        chart_height_spin.setValue(self.config.get('chart_height', 200))
        chart_height_spin.setSingleStep(10)
        chart_height_spin.setFixedWidth(70)
        chart_height_spin.valueChanged.connect(self.sync_shared_controls)
        layout.addWidget(chart_height_spin)
        
        layout.addStretch()
        
        # Register controls
        self.shared_controls.append({
            'file_label': file_name_label,
            'start': start_time_input,
            'dur': duration_input,
            'fps': fps_input,
            'rows_label': total_rows_label,
            'len_label': signal_length_label,
            'chart_height': chart_height_spin
        })
        
        # Initialize labels if file loaded
        if self.selected_file_path:
             file_name_label.setText(os.path.basename(self.selected_file_path))
             if self.signal_data is not None:
                 total_rows_label.setText(f"Rows: {len(self.signal_data)}")
                 # Trigger update to set length label
                 # We can't call sync here easily without recursion risk or partial init, 
                 # but update_signal_info logic handles it.
        
        return container

    def sync_shared_controls(self):
        """Syncs all signal control widgets and updates config"""
        sender = self.sender()
        
        # Find values from the sender or config
        start = self.config.get('signal', {}).get('start_time', 1.0)
        duration = self.config.get('signal', {}).get('duration', 10.0)
        fps = self.config['fps']
        chart_height = self.config.get('chart_height', 200)

        if isinstance(sender, (QDoubleSpinBox, QSpinBox)):
            # Identify which parameter changed
            for controls in self.shared_controls:
                if sender == controls['start']:
                    start = sender.value()
                    break
                elif sender == controls['dur']:
                    duration = sender.value()
                    break
                elif sender == controls['fps']:
                    fps = sender.value()
                    break
                elif sender == controls['chart_height']:
                    chart_height = sender.value()
                    break
        
        # Update config
        self.config['fps'] = fps
        self.config['chart_height'] = chart_height
        if 'signal' not in self.config: self.config['signal'] = {}
        self.config['signal']['start_time'] = start
        self.config['signal']['duration'] = duration
        
        # Update all widgets
        start_idx = int(start * fps)
        length = int(duration * fps)
        len_text = f"Len: {length} (idx {start_idx} to {start_idx + length})"
        
        for controls in self.shared_controls:
            controls['start'].blockSignals(True)
            controls['dur'].blockSignals(True)
            controls['fps'].blockSignals(True)
            controls['chart_height'].blockSignals(True)
            
            controls['start'].setValue(start)
            controls['dur'].setValue(duration)
            controls['fps'].setValue(fps)
            controls['chart_height'].setValue(chart_height)
            controls['len_label'].setText(len_text)
            
            controls['start'].blockSignals(False)
            controls['dur'].blockSignals(False)
            controls['fps'].blockSignals(False)
            controls['chart_height'].blockSignals(False)
            
        # Trigger chart height update
        self.update_chart_heights()

    def create_preprocessing_controls(self):
        group = QGroupBox("Preprocessing")
        group.setStyleSheet("QGroupBox { border: 2px solid #a0aec0; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; }")
        lay = QGridLayout(group)
        lay.setSpacing(4) # Reduced spacing
        lay.setContentsMargins(4, 4, 4, 4) # Reduced margins
        
        # Detrend
        self.detrend_check = QCheckBox("Detrend")
        self.detrend_check.setChecked(self.config.get('preprocessing', {}).get('detrend', {}).get('enabled', True))
        self.detrend_check.setToolTip("Enable or disable detrending of the signal.")
        lay.addWidget(self.detrend_check, 0, 0)
        
        self.detrend_lambda = QSpinBox()
        self.detrend_lambda.setRange(10, 10000)
        self.detrend_lambda.setValue(self.config.get('preprocessing', {}).get('detrend', {}).get('lambda', 100))
        self.detrend_lambda.setSingleStep(10)
        self.detrend_lambda.setToolTip("Lambda parameter for detrending. Higher values result in smoother trends.")
        lay.addWidget(self.detrend_lambda, 0, 1)
        
        # Harmonics
        self.harmonics_check = QCheckBox("Harmonics")
        self.harmonics_check.setChecked(self.config.get('preprocessing', {}).get('harmonics', {}).get('enabled', False))
        self.harmonics_check.setToolTip("Enable or disable harmonic enhancement.")
        lay.addWidget(self.harmonics_check, 1, 0)
        
        self.harmonics_gain = QDoubleSpinBox()
        self.harmonics_gain.setRange(1.0, 10.0)
        self.harmonics_gain.setValue(self.config.get('preprocessing', {}).get('harmonics', {}).get('harmonics_gain', 2.0))
        self.harmonics_gain.setSingleStep(0.1)
        self.harmonics_gain.setToolTip("Gain factor applied to harmonic frequencies.")
        lay.addWidget(self.harmonics_gain, 1, 1)
        
        # FFT Size
        lay.addWidget(QLabel("FFT Size:"), 2, 0)
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.fft_size_combo.setCurrentText(str(self.config.get('preprocessing', {}).get('fft_size', 1024)))
        self.fft_size_combo.setToolTip("Size of the FFT window used for spectrogram and PSD calculations.")
        lay.addWidget(self.fft_size_combo, 2, 1)
        
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Added
        return group

    def create_vmd_param_controls(self):
        group = QGroupBox("VMD Parameters")
        group.setStyleSheet("QGroupBox { border: 2px solid #a0aec0; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; }")
        lay = QGridLayout(group)
        lay.setSpacing(4) # Reduced spacing
        lay.setContentsMargins(4, 4, 4, 4) # Reduced margins
        self.vmd_inputs = {}
        params = [("K", 2, 50, self.config['vmd']['K'], True, "Number of VMD modes to extract."),
                  ("Alpha", 1, 10000, self.config['vmd']['alpha'], False, "Bandwidth constraint parameter (alpha). Controls the fidelity of the reconstruction."),
                  ("Tau", 0, 1, self.config['vmd']['tau'], False, "Noise-tolerance (tau). Controls the dual ascent step size."),
                  ("DC", 0, 1, self.config['vmd']['DC'], True, "Determines if DC part is kept (0) or removed (1)."),
                  ("Init", 0, 2, self.config['vmd']['init'], True, "Initialization method for VMD modes (0: all zeros, 1: random, 2: user-defined)."),
                  ("Tol", 1e-10, 1e-3, self.config['vmd']['tol'], False, "Tolerance for convergence criterion.")]
        
        row = 0
        col = 0
        for name, min_val, max_val, default, is_int, tooltip in params:
            label = QLabel(f"{name}:")
            if is_int:
                spinbox = QSpinBox()
                spinbox.setRange(int(min_val), int(max_val))
                spinbox.setValue(int(default))
            else:
                spinbox = QDoubleSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setValue(default)
                if name == "Tol": spinbox.setDecimals(7)
            
            spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spinbox.setToolTip(tooltip)
            self.vmd_inputs[name.lower()] = spinbox
            
            lay.addWidget(label, row, col)
            lay.addWidget(spinbox, row, col + 1)
            
            col += 2
            if col >= 4: # 2 pairs per row
                col = 0
                row += 1
                
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Added
        return group

    def create_mode_selection_controls(self):
        group = QGroupBox("Mode Selection")
        group.setStyleSheet("QGroupBox { border: 2px solid #a0aec0; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; }")
        lay = QGridLayout(group)
        lay.setSpacing(4) # Reduced spacing
        lay.setContentsMargins(4, 4, 4, 4) # Reduced margins
        self.mode_inputs = {}
        
        # Freq Range
        lay.addWidget(QLabel("Freq Range (Hz):"), 0, 0)
        hbox_freq = QHBoxLayout()
        hbox_freq.setSpacing(2) # Tight spacing
        self.mode_inputs['freq_min'] = QDoubleSpinBox()
        self.mode_inputs['freq_min'].setSingleStep(0.1) # Set step to 0.1
        self.mode_inputs['freq_min'].setValue(self.config['mode_selection']['freq_min'])
        self.mode_inputs['freq_min'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_inputs['freq_min'].setToolTip("Minimum frequency (Hz) for mode selection.")
        hbox_freq.addWidget(self.mode_inputs['freq_min'])
        
        hbox_freq.addWidget(QLabel("-"))
        
        self.mode_inputs['freq_max'] = QDoubleSpinBox()
        self.mode_inputs['freq_max'].setSingleStep(0.1) # Set step to 0.1
        self.mode_inputs['freq_max'].setValue(self.config['mode_selection']['freq_max'])
        self.mode_inputs['freq_max'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_inputs['freq_max'].setToolTip("Maximum frequency (Hz) for mode selection.")
        hbox_freq.addWidget(self.mode_inputs['freq_max'])
        lay.addLayout(hbox_freq, 0, 1, 1, 3)

        # Correlation
        lay.addWidget(QLabel("Corr >"), 1, 0)
        self.mode_inputs['correlation_threshold'] = QDoubleSpinBox()
        self.mode_inputs['correlation_threshold'].setValue(self.config['mode_selection']['correlation_threshold'])
        self.mode_inputs['correlation_threshold'].setSingleStep(0.05)
        self.mode_inputs['correlation_threshold'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.mode_inputs['correlation_threshold'], 1, 1)

        # Energy
        lay.addWidget(QLabel("Energy % >"), 1, 2)
        self.mode_inputs['energy_threshold'] = QDoubleSpinBox()
        self.mode_inputs['energy_threshold'].setValue(self.config['mode_selection']['energy_threshold'])
        self.mode_inputs['energy_threshold'].setSingleStep(0.1) # Keep 0.1 for percentage
        self.mode_inputs['energy_threshold'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.mode_inputs['energy_threshold'], 1, 3)

        # Kurtosis
        lay.addWidget(QLabel("Kurtosis <"), 2, 0)
        self.mode_inputs['kurtosis_max'] = QDoubleSpinBox()
        self.mode_inputs['kurtosis_max'].setValue(self.config['mode_selection']['kurtosis_max'])
        self.mode_inputs['kurtosis_max'].setSingleStep(0.1) # Keep 0.1
        self.mode_inputs['kurtosis_max'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.mode_inputs['kurtosis_max'], 2, 1)

        # Method
        lay.addWidget(QLabel("Method:"), 2, 2)
        self.selection_method_combo = QComboBox()
        self.selection_method_combo.addItems(["frequency_only", "frequency+correlation", "all_criteria", "frequency+harmonics"])
        self.selection_method_combo.setCurrentText(self.config['mode_selection']['selection_method'])
        self.selection_method_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.selection_method_combo, 2, 3)

        # Ref
        lay.addWidget(QLabel("Corr Ref:"), 3, 0)
        self.corr_ref_combo = QComboBox()
        self.corr_ref_combo.addItems(["original", "bandpass", "selected"])
        self.corr_ref_combo.setCurrentText(self.config['mode_selection']['correlation_reference'])
        self.corr_ref_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.corr_ref_combo, 3, 1, 1, 3)
        
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Added
        return group

    def create_optimization_controls(self):
        group = QGroupBox("Optimization")
        group.setStyleSheet("QGroupBox { border: 2px solid #a0aec0; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; }")
        lay = QGridLayout(group)
        lay.setSpacing(4) # Reduced spacing
        lay.setContentsMargins(4, 4, 4, 4) # Reduced margins
        
        # K Range
        lay.addWidget(QLabel("K Range:"), 0, 0)
        hbox_k = QHBoxLayout()
        hbox_k.setSpacing(2) # Tight spacing
        self.k_min_input = QSpinBox()
        self.k_min_input.setRange(2, 20)
        self.k_min_input.setValue(self.config['auto_optimize']['K_range'][0])
        self.k_min_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.k_min_input)
        
        hbox_k.addWidget(QLabel("-"))
        
        self.k_max_input = QSpinBox()
        self.k_max_input.setRange(2, 20)
        self.k_max_input.setValue(self.config['auto_optimize']['K_range'][1])
        self.k_max_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hbox_k.addWidget(self.k_max_input)
        lay.addLayout(hbox_k, 0, 1)
        
        # Alpha Range
        lay.addWidget(QLabel("Alpha Range:"), 1, 0)
        hbox_alpha = QHBoxLayout()
        hbox_alpha.setSpacing(2) # Tight spacing
        self.alpha_min_input = QDoubleSpinBox()
        self.alpha_min_input.setRange(1, 20000)
        self.alpha_min_input.setSingleStep(100)
        self.alpha_min_input.setValue(self.config['auto_optimize']['alpha_range'][0])
        self.alpha_min_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hbox_alpha.addWidget(self.alpha_min_input)
        
        hbox_alpha.addWidget(QLabel("-"))
        
        self.alpha_max_input = QDoubleSpinBox()
        self.alpha_max_input.setRange(1, 20000)
        self.alpha_max_input.setSingleStep(100)
        self.alpha_max_input.setValue(self.config['auto_optimize']['alpha_range'][1])
        self.alpha_max_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hbox_alpha.addWidget(self.alpha_max_input)
        lay.addLayout(hbox_alpha, 1, 1)
        
        # Metric
        lay.addWidget(QLabel("Metric:"), 2, 0)
        self.opt_metric_combo = QComboBox()
        self.opt_metric_combo.addItems(["snr", "spectral_purity", "combined"])
        self.opt_metric_combo.setCurrentText(self.config['auto_optimize']['optimization_metric'])
        self.opt_metric_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.opt_metric_combo, 2, 1)
        
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Added
        return group

    def create_vmd_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Buttons on top
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0,0,0,0)
        button_layout.setSpacing(10)
        
        self.run_vmd_btn = QPushButton("▶ Run VMD")
        self.run_vmd_btn.clicked.connect(self.run_vmd)
        self.run_vmd_btn.setStyleSheet("QPushButton { background-color: #10b981; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; font-size: 12px; } QPushButton:hover { background-color: #059669; } QPushButton:pressed { background-color: #047857; }")
        
        self.show_imfs_btn = QPushButton("Show IMFs")
        self.show_imfs_btn.setCheckable(True)
        self.show_imfs_btn.setChecked(self.config['ui_state'].get('show_individual_modes', False))
        self.show_imfs_btn.toggled.connect(self.on_show_imfs_toggled)
        self.show_imfs_btn.setStyleSheet("QPushButton { background-color: #6b7280; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; font-size: 12px; } QPushButton:checked { background-color: #4b5563; } QPushButton:hover { background-color: #4b5563; }")

        self.auto_opt_btn = QPushButton("🔍 Auto-Optimize")
        self.auto_opt_btn.clicked.connect(self.run_auto_optimize)
        self.auto_opt_btn.setStyleSheet("QPushButton { background-color: #8b5cf6; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; font-size: 12px; } QPushButton:hover { background-color: #7c3aed; } QPushButton:pressed { background-color: #6d28d9; }")
        
        # Config buttons style
        btn_style = "QPushButton { background-color: #3b82f6; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; font-size: 12px; } QPushButton:hover { background-color: #2563eb; } QPushButton:pressed { background-color: #1d4ed8; }"

        # Separator for config buttons
        config_btn_separator = QFrame()
        config_btn_separator.setFrameShape(QFrame.VLine)
        config_btn_separator.setFrameShadow(QFrame.Sunken)
        
        self.save_data_btn = QPushButton("💾 Save Data")
        self.save_data_btn.clicked.connect(self.open_save_data_dialog)
        self.save_data_btn.setStyleSheet(btn_style)

        self.load_config_btn = QPushButton("📂 Load Config")
        self.load_config_btn.clicked.connect(self.load_custom_config)
        self.load_config_btn.setStyleSheet(btn_style)

        self.save_config_btn = QPushButton("💾 Save Config")
        self.save_config_btn.clicked.connect(self.save_current_config)
        self.save_config_btn.setStyleSheet(btn_style)

        self.default_config_btn = QPushButton("🔄 Default Config")
        self.default_config_btn.clicked.connect(self.load_default_config)
        self.default_config_btn.setStyleSheet(btn_style)

        # Progress Bar
        self.opt_progress = QProgressBar()
        self.opt_progress.setRange(0, 100)
        self.opt_progress.setTextVisible(True)
        self.opt_progress.setFormat("%p%")
        self.opt_progress.setVisible(False)
        self.opt_progress.setFixedWidth(200)
        self.opt_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #a0aec0;
                border-radius: 5px;
                text-align: center;
                background-color: #e2e8f0;
                height: 22px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #8b5cf6;
                border-radius: 4px;
            }
        """)

        # Result Label
        self.opt_result_label = QLabel("")
        self.opt_result_label.setStyleSheet("color: #2563eb; font-weight: bold; margin-left: 10px;")

        button_layout.addWidget(self.run_vmd_btn)
        button_layout.addWidget(self.show_imfs_btn)
        button_layout.addWidget(self.auto_opt_btn)
        button_layout.addWidget(config_btn_separator)
        button_layout.addWidget(self.save_data_btn)
        button_layout.addWidget(self.load_config_btn)
        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.default_config_btn)
        button_layout.addWidget(self.opt_progress)
        button_layout.addWidget(self.opt_result_label)
        button_layout.addStretch()
        
        main_layout.addWidget(button_container)

        # Collapsible Configuration Box
        config_box = CollapsibleBox("Configuration")
        config_content = QWidget()
        
        grid = QGridLayout(config_content)
        grid.setSpacing(6)
        grid.setContentsMargins(4,4,4,4)
        grid.setColumnStretch(1, 1)

        # Subject Info (Shared)
        grid.addWidget(self.create_shared_info_widget(), 0, 0, 1, 4)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        grid.addWidget(line, 1, 0, 1, 4)

        # VMD Controls in a single row (4 columns)
        grid.addWidget(self.create_preprocessing_controls(), 2, 0)
        grid.addWidget(self.create_vmd_param_controls(), 2, 1)
        grid.addWidget(self.create_mode_selection_controls(), 2, 2)
        grid.addWidget(self.create_optimization_controls(), 2, 3)
        
        # Set column stretches to be equal
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        
        config_box.setContentLayout(grid)
        
        main_layout.addWidget(config_box)

        return tab

    def on_show_imfs_toggled(self, checked):
        self.config['ui_state']['show_individual_modes'] = checked
        if self.vmd_results:
            self.display_vmd_results()

    def create_comparison_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(4, 4, 4, 4)

        config_box = CollapsibleBox("Configuration")
        config_content = QWidget()
        
        layout = QVBoxLayout(config_content)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Subject Info (Shared)
        layout.addWidget(self.create_shared_info_widget())
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Filter Settings
        # Reorganized into two rows: 
        # Row 1 (Complex): Elliptic, Chebyshev I, Chebyshev II (3 items)
        # Row 2 (Simpler): Butterworth, Harmonics, Moving Average, Savitzky-Golay, Wavelet (5 items)
        
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)
        
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)

        # Styles for distinct colors
        style_butter = "QGroupBox { border: 2px solid #3b82f6; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #1e40af; }"
        style_cheby = "QGroupBox { border: 2px solid #ef4444; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #991b1b; }"
        style_cheby2 = "QGroupBox { border: 2px solid #f97316; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #9a3412; }"
        style_elliptic = "QGroupBox { border: 2px solid #84cc16; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #4d7c0f; }"
        style_ma = "QGroupBox { border: 2px solid #10b981; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #065f46; }"
        style_savgol = "QGroupBox { border: 2px solid #8b5cf6; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #5b21b6; }"
        style_harmonics = "QGroupBox { border: 2px solid #d97706; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #b45309; }"
        style_wavelet = "QGroupBox { border: 2px solid #0ea5e9; border-radius: 4px; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; font-weight: bold; font-size: 11pt; color: #0369a1; }"

        # Butterworth
        butter_group = QGroupBox("Butterworth")
        butter_group.setStyleSheet(style_butter)
        butter_group.setCheckable(True)
        butter_group.setChecked(self.config['traditional_filters']['butterworth'].get('enabled', True))
        self.butter_check = butter_group
        
        b_lay = QGridLayout(butter_group)
        self.butter_inputs = {}
        b_lay.addWidget(QLabel("Order:"), 0, 0)
        self.butter_inputs['order'] = QSpinBox()
        self.butter_inputs['order'].setRange(1, 10)
        self.butter_inputs['order'].setValue(self.config['traditional_filters']['butterworth']['order'])
        self.butter_inputs['order'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b_lay.addWidget(self.butter_inputs['order'], 0, 1)
        b_lay.addWidget(QLabel("F Min:"), 0, 2)
        self.butter_inputs['freq_min'] = QDoubleSpinBox()
        self.butter_inputs['freq_min'].setRange(0.1, 10)
        self.butter_inputs['freq_min'].setSingleStep(0.1)
        self.butter_inputs['freq_min'].setValue(self.config['traditional_filters']['butterworth']['freq_min'])
        self.butter_inputs['freq_min'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b_lay.addWidget(self.butter_inputs['freq_min'], 0, 3)
        b_lay.addWidget(QLabel("F Max:"), 0, 4)
        self.butter_inputs['freq_max'] = QDoubleSpinBox()
        self.butter_inputs['freq_max'].setRange(0.1, 10)
        self.butter_inputs['freq_max'].setSingleStep(0.1)
        self.butter_inputs['freq_max'].setValue(self.config['traditional_filters']['butterworth']['freq_max'])
        self.butter_inputs['freq_max'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b_lay.addWidget(self.butter_inputs['freq_max'], 0, 5)
        b_lay.setColumnStretch(1, 1)
        b_lay.setColumnStretch(3, 1)
        b_lay.setColumnStretch(5, 1)
        butter_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row2_layout.addWidget(butter_group)

        # Chebyshev I
        cheby_group = QGroupBox("Chebyshev I")
        cheby_group.setStyleSheet(style_cheby)
        cheby_group.setCheckable(True)
        cheby_group.setChecked(self.config['traditional_filters']['chebyshev'].get('enabled', True))
        self.cheby_check = cheby_group
        
        c_lay = QGridLayout(cheby_group)
        self.cheby_inputs = {}
        c_lay.addWidget(QLabel("Order:"), 0, 0)
        self.cheby_inputs['order'] = QSpinBox()
        self.cheby_inputs['order'].setRange(1, 10)
        self.cheby_inputs['order'].setValue(self.config['traditional_filters']['chebyshev']['order'])
        self.cheby_inputs['order'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c_lay.addWidget(self.cheby_inputs['order'], 0, 1)
        c_lay.addWidget(QLabel("Ripple:"), 0, 2)
        self.cheby_inputs['ripple'] = QDoubleSpinBox()
        self.cheby_inputs['ripple'].setRange(0.1, 5)
        self.cheby_inputs['ripple'].setSingleStep(0.1)
        self.cheby_inputs['ripple'].setValue(self.config['traditional_filters']['chebyshev']['ripple'])
        self.cheby_inputs['ripple'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c_lay.addWidget(self.cheby_inputs['ripple'], 0, 3)
        c_lay.addWidget(QLabel("F Min:"), 1, 0)
        self.cheby_inputs['freq_min'] = QDoubleSpinBox()
        self.cheby_inputs['freq_min'].setSingleStep(0.1)
        self.cheby_inputs['freq_min'].setValue(self.config['traditional_filters']['chebyshev']['freq_min'])
        self.cheby_inputs['freq_min'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c_lay.addWidget(self.cheby_inputs['freq_min'], 1, 1)
        c_lay.addWidget(QLabel("F Max:"), 1, 2)
        self.cheby_inputs['freq_max'] = QDoubleSpinBox()
        self.cheby_inputs['freq_max'].setSingleStep(0.1)
        self.cheby_inputs['freq_max'].setValue(self.config['traditional_filters']['chebyshev']['freq_max'])
        self.cheby_inputs['freq_max'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c_lay.addWidget(self.cheby_inputs['freq_max'], 1, 3)
        c_lay.setColumnStretch(1, 1)
        c_lay.setColumnStretch(3, 1)
        cheby_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row1_layout.addWidget(cheby_group)

        # Chebyshev II
        cheby2_group = QGroupBox("Chebyshev II")
        cheby2_group.setStyleSheet(style_cheby2)
        cheby2_group.setCheckable(True)
        cheby2_group.setChecked(self.config['traditional_filters']['cheby2'].get('enabled', True))
        self.cheby2_check = cheby2_group

        c2_lay = QGridLayout(cheby2_group)
        self.cheby2_inputs = {}
        c2_lay.addWidget(QLabel("Order:"), 0, 0)
        self.cheby2_inputs['order'] = QSpinBox()
        self.cheby2_inputs['order'].setRange(1, 10)
        self.cheby2_inputs['order'].setValue(self.config['traditional_filters']['cheby2']['order'])
        self.cheby2_inputs['order'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c2_lay.addWidget(self.cheby2_inputs['order'], 0, 1)
        c2_lay.addWidget(QLabel("Stop Atten:"), 0, 2)
        self.cheby2_inputs['stopband_atten'] = QSpinBox()
        self.cheby2_inputs['stopband_atten'].setRange(20, 100)
        self.cheby2_inputs['stopband_atten'].setValue(self.config['traditional_filters']['cheby2']['stopband_atten'])
        self.cheby2_inputs['stopband_atten'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c2_lay.addWidget(self.cheby2_inputs['stopband_atten'], 0, 3)
        c2_lay.addWidget(QLabel("F Min:"), 1, 0)
        self.cheby2_inputs['freq_min'] = QDoubleSpinBox()
        self.cheby2_inputs['freq_min'].setSingleStep(0.1)
        self.cheby2_inputs['freq_min'].setValue(self.config['traditional_filters']['cheby2']['freq_min'])
        self.cheby2_inputs['freq_min'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c2_lay.addWidget(self.cheby2_inputs['freq_min'], 1, 1)
        c2_lay.addWidget(QLabel("F Max:"), 1, 2)
        self.cheby2_inputs['freq_max'] = QDoubleSpinBox()
        self.cheby2_inputs['freq_max'].setSingleStep(0.1)
        self.cheby2_inputs['freq_max'].setValue(self.config['traditional_filters']['cheby2']['freq_max'])
        self.cheby2_inputs['freq_max'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c2_lay.addWidget(self.cheby2_inputs['freq_max'], 1, 3)
        c2_lay.setColumnStretch(1, 1)
        c2_lay.setColumnStretch(3, 1)
        cheby2_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row1_layout.addWidget(cheby2_group)

        # Harmonics
        harmonics_group = QGroupBox("Harmonics Enhancement")
        harmonics_group.setStyleSheet(style_harmonics)
        harmonics_group.setCheckable(True)
        harmonics_group.setChecked(self.config['traditional_filters']['harmonics'].get('enabled', True))
        self.filter_harmonics_check = harmonics_group
        
        h_lay = QHBoxLayout(harmonics_group)
        self.filter_harmonics_gain = QDoubleSpinBox()
        self.filter_harmonics_gain.setRange(1.0, 10.0)
        self.filter_harmonics_gain.setSingleStep(0.1)
        self.filter_harmonics_gain.setValue(self.config.get('traditional_filters', {}).get('harmonics', {}).get('harmonics_gain', 2.0))
        self.filter_harmonics_gain.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_lay.addWidget(QLabel("Gain:"))
        h_lay.addWidget(self.filter_harmonics_gain)
        h_lay.addStretch()
        harmonics_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row2_layout.addWidget(harmonics_group)

        # Elliptic
        elliptic_group = QGroupBox("Elliptic")
        elliptic_group.setStyleSheet(style_elliptic)
        elliptic_group.setCheckable(True)
        elliptic_group.setChecked(self.config['traditional_filters']['elliptic'].get('enabled', True))
        self.elliptic_check = elliptic_group

        e_lay = QGridLayout(elliptic_group)
        self.elliptic_inputs = {}
        e_lay.addWidget(QLabel("Order:"), 0, 0)
        self.elliptic_inputs['order'] = QSpinBox()
        self.elliptic_inputs['order'].setRange(1, 10)
        self.elliptic_inputs['order'].setValue(self.config['traditional_filters']['elliptic']['order'])
        self.elliptic_inputs['order'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        e_lay.addWidget(self.elliptic_inputs['order'], 0, 1)
        e_lay.addWidget(QLabel("PB Ripple:"), 0, 2)
        self.elliptic_inputs['passband_ripple'] = QDoubleSpinBox()
        self.elliptic_inputs['passband_ripple'].setRange(0.1, 5)
        self.elliptic_inputs['passband_ripple'].setSingleStep(0.1)
        self.elliptic_inputs['passband_ripple'].setValue(self.config['traditional_filters']['elliptic']['passband_ripple'])
        self.elliptic_inputs['passband_ripple'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        e_lay.addWidget(self.elliptic_inputs['passband_ripple'], 0, 3)
        e_lay.addWidget(QLabel("SB Atten:"), 0, 4)
        self.elliptic_inputs['stopband_atten'] = QSpinBox()
        self.elliptic_inputs['stopband_atten'].setRange(20, 100)
        self.elliptic_inputs['stopband_atten'].setValue(self.config['traditional_filters']['elliptic']['stopband_atten'])
        self.elliptic_inputs['stopband_atten'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        e_lay.addWidget(self.elliptic_inputs['stopband_atten'], 0, 5)
        e_lay.addWidget(QLabel("F Min:"), 1, 0)
        self.elliptic_inputs['freq_min'] = QDoubleSpinBox()
        self.elliptic_inputs['freq_min'].setSingleStep(0.1)
        self.elliptic_inputs['freq_min'].setValue(self.config['traditional_filters']['elliptic']['freq_min'])
        self.elliptic_inputs['freq_min'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        e_lay.addWidget(self.elliptic_inputs['freq_min'], 1, 1)
        e_lay.addWidget(QLabel("F Max:"), 1, 2)
        self.elliptic_inputs['freq_max'] = QDoubleSpinBox()
        self.elliptic_inputs['freq_max'].setSingleStep(0.1)
        self.elliptic_inputs['freq_max'].setValue(self.config['traditional_filters']['elliptic']['freq_max'])
        self.elliptic_inputs['freq_max'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        e_lay.addWidget(self.elliptic_inputs['freq_max'], 1, 3)
        e_lay.setColumnStretch(1, 1)
        e_lay.setColumnStretch(3, 1)
        e_lay.setColumnStretch(5, 1)
        elliptic_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row1_layout.addWidget(elliptic_group)

        # Moving Average
        ma_group = QGroupBox("Moving Average")
        ma_group.setStyleSheet(style_ma)
        ma_group.setCheckable(True)
        ma_group.setChecked(self.config['traditional_filters']['moving_average'].get('enabled', True))
        self.ma_check = ma_group
        
        m_lay = QHBoxLayout(ma_group)
        self.ma_inputs = {}
        m_lay.addWidget(QLabel("Window Size:"))
        self.ma_inputs['window_size'] = QSpinBox()
        self.ma_inputs['window_size'].setRange(3, 101)
        self.ma_inputs['window_size'].setSingleStep(2)
        self.ma_inputs['window_size'].setValue(self.config['traditional_filters']['moving_average']['window_size'])
        self.ma_inputs['window_size'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        m_lay.addWidget(self.ma_inputs['window_size'])
        m_lay.addStretch()
        ma_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row2_layout.addWidget(ma_group)

        # Savitzky-Golay
        savgol_group = QGroupBox("Savitzky-Golay")
        savgol_group.setStyleSheet(style_savgol)
        savgol_group.setCheckable(True)
        savgol_group.setChecked(self.config['traditional_filters']['savgol'].get('enabled', True))
        self.savgol_check = savgol_group
        
        s_lay = QGridLayout(savgol_group) # Changed to QGridLayout
        self.savgol_inputs = {}
        s_lay.addWidget(QLabel("Window:"), 0, 0)
        self.savgol_inputs['window_size'] = QSpinBox()
        self.savgol_inputs['window_size'].setRange(5, 101)
        self.savgol_inputs['window_size'].setSingleStep(2)
        self.savgol_inputs['window_size'].setValue(self.config['traditional_filters']['savgol']['window_size'])
        self.savgol_inputs['window_size'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        s_lay.addWidget(self.savgol_inputs['window_size'], 0, 1)
        s_lay.addWidget(QLabel("Poly:"), 1, 0)
        self.savgol_inputs['poly_order'] = QSpinBox()
        self.savgol_inputs['poly_order'].setRange(1, 10)
        self.savgol_inputs['poly_order'].setValue(self.config['traditional_filters']['savgol']['poly_order'])
        self.savgol_inputs['poly_order'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        s_lay.addWidget(self.savgol_inputs['poly_order'], 1, 1)
        savgol_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row1_layout.addWidget(savgol_group) # Moved to row 1

        # Wavelet
        wavelet_group = QGroupBox("Wavelet Denoising")
        wavelet_group.setStyleSheet(style_wavelet)
        wavelet_group.setCheckable(True)
        wavelet_group.setChecked(self.config['traditional_filters'].get('wavelet', {}).get('enabled', True))
        self.wavelet_check = wavelet_group
        
        w_lay = QHBoxLayout(wavelet_group)
        self.wavelet_inputs = {}
        w_lay.addWidget(QLabel("Wavelet:"))
        self.wavelet_inputs['wavelet'] = QComboBox()
        self.wavelet_inputs['wavelet'].addItems(['db4', 'sym4', 'coif4', 'haar', 'db8', 'sym8'])
        self.wavelet_inputs['wavelet'].setCurrentText(self.config['traditional_filters'].get('wavelet', {}).get('wavelet', 'db4'))
        self.wavelet_inputs['wavelet'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        w_lay.addWidget(self.wavelet_inputs['wavelet'])
        
        w_lay.addWidget(QLabel("Level:"))
        self.wavelet_inputs['level'] = QSpinBox()
        self.wavelet_inputs['level'].setRange(1, 10)
        self.wavelet_inputs['level'].setValue(self.config['traditional_filters']['wavelet']['level'])
        self.wavelet_inputs['level'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        w_lay.addWidget(self.wavelet_inputs['level'])
        
        w_lay.addWidget(QLabel("Mode:"))
        self.wavelet_inputs['threshold_mode'] = QComboBox()
        self.wavelet_inputs['threshold_mode'].addItems(['soft', 'hard'])
        self.wavelet_inputs['threshold_mode'].setCurrentText(self.config['traditional_filters'].get('wavelet', {}).get('threshold_mode', 'soft'))
        self.wavelet_inputs['threshold_mode'].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        w_lay.addWidget(self.wavelet_inputs['threshold_mode'])
        w_lay.addStretch()
        wavelet_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row2_layout.addWidget(wavelet_group)

        layout.addLayout(row1_layout)
        layout.addLayout(row2_layout)
        layout.addStretch()
        
        config_box.setContentLayout(layout)
        main_layout.addWidget(config_box)

        # Run Button
        self.run_filters_btn = QPushButton("▶ Run Filters")
        self.run_filters_btn.clicked.connect(self.run_all_filters)
        self.run_filters_btn.setStyleSheet("QPushButton { background-color: #10b981; color: white; font-weight: bold; padding: 8px; border-radius: 4px; font-size: 14px; } QPushButton:hover { background-color: #059669; } QPushButton:pressed { background-color: #047857; }")
        
        main_layout.addWidget(self.run_filters_btn)
        main_layout.addStretch()

        return tab

    def update_chart_heights(self):
        height = self.config.get('chart_height', 200)
        # Update VMD charts
        for canvas in self.vmd_chart_rows:
            canvas.setMinimumHeight(height)
            canvas.updateGeometry()
        # Update Filter charts
        for canvas in self.filter_chart_rows:
            canvas.setMinimumHeight(height)
            canvas.updateGeometry()
        # Update Pipeline charts
        for canvas in self.pipeline_chart_rows:
            canvas.setMinimumHeight(height)
            canvas.draw()
            canvas.updateGeometry()

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory", self.current_directory)
        if directory:
            self.current_directory = directory
            self.load_csv_files(directory)

    def load_csv_files(self, directory):
        self.file_list.clear()
        self.path_label.setText(directory)
        try:
            files = [f for f in os.listdir(directory) if f.endswith('.csv')]
            self.file_list.addItems(files)
        except Exception as e:
            print(f"Error loading files: {e}")
            traceback.print_exc()

    def on_file_selected(self, item):
        filename = item.text()
        filepath = os.path.join(self.current_directory, filename)
        try:
            df = pd.read_csv(filepath)
            col_name = self.config['rppg_column_name']
            if col_name not in df.columns:
                print(f"Column '{col_name}' not found in CSV.")
                return
            self.signal_data = df[col_name].values
            self.selected_file_path = filepath
            
            # Update all file labels
            for controls in self.shared_controls:
                controls['file_label'].setText(filename)
                controls['rows_label'].setText(f"Rows: {len(df)}")
            
            # Trigger sync to update length labels
            self.sync_shared_controls()
            
        except Exception as e:
            print(f"Failed to load file: {e}")
            traceback.print_exc()

    def get_selected_signal_segment(self):
        if self.signal_data is None: return None
        fps = self.config['fps']
        start = self.config['signal']['start_time']
        duration = self.config['signal']['duration']
        
        start_idx, end_idx = int(start * fps), int(start * fps) + int(duration * fps)
        if end_idx > len(self.signal_data):
            print("Warning: Selected segment exceeds signal length. Using available data.")
            end_idx = len(self.signal_data)
        return self.signal_data[start_idx:end_idx]

    def run_vmd(self):
        signal = self.get_selected_signal_segment()
        if signal is None or len(signal) == 0:
            QMessageBox.warning(self, "No Signal Selected", "Please select a CSV file from the list to analyze.")
            return
        self.run_vmd_btn.setEnabled(False)
        self.run_vmd_btn.setText("Processing...")
        QApplication.processEvents()
        
        params = {
            'signal': signal,
            'fps': self.config['fps'],
            'detrend': {'enabled': self.detrend_check.isChecked(), 'lambda': self.detrend_lambda.value()},
            'normalize': {'enabled': True, 'method': 'z-score'}, # Force enabled
            'harmonics': {
                'enabled': self.harmonics_check.isChecked(),
                'gain': self.harmonics_gain.value(),
                'freq_min': self.mode_inputs['freq_min'].value(),
                'freq_max': self.mode_inputs['freq_max'].value()
            },
            'fft_size': int(self.fft_size_combo.currentText()),
            'vmd_params': {k: v.value() for k, v in self.vmd_inputs.items()},
            'selection_params': {k: v.value() for k, v in self.mode_inputs.items()},
        }
        params['selection_params'].update({'selection_method': self.selection_method_combo.currentText(), 'correlation_reference': self.corr_ref_combo.currentText()})
        
        self.vmd_thread = ComputationThread(run_vmd_computation, params)
        self.vmd_thread.finished.connect(self.on_vmd_finished)
        self.vmd_thread.error.connect(self.on_computation_error)
        self.vmd_thread.start()

    def on_vmd_finished(self, results):
        self.vmd_results = results
        self.display_vmd_results()
        self.run_vmd_btn.setEnabled(True)
        self.run_vmd_btn.setText("▶ Run VMD")

    def on_computation_error(self, error_msg):
        QMessageBox.critical(self, "Computation Error", f"An error occurred:\n{error_msg}")
        self.run_vmd_btn.setEnabled(True)
        self.run_vmd_btn.setText("▶ Run VMD")
        self.run_filters_btn.setEnabled(True)
        self.run_filters_btn.setText("▶ Run Filters")

    def run_auto_optimize(self):
        signal = self.get_selected_signal_segment()
        if signal is None or len(signal) == 0:
            QMessageBox.warning(self, "No Signal Selected", "Please select a CSV file from the list to analyze.")
            return
        if self.detrend_check.isChecked(): signal = computation.detrend_signal(signal, self.detrend_lambda.value())
        # Always normalize
        signal = computation.normalize_signal(signal, 'z-score')
        
        if self.harmonics_check.isChecked():
            gain = self.harmonics_gain.value()
            freq_min = self.mode_inputs['freq_min'].value()
            freq_max = self.mode_inputs['freq_max'].value()
            signal = computation.enhance_harmonics(signal, self.config['fps'], freq_min, freq_max, gain)
        
        fps = self.config['fps']
        K_range = [self.k_min_input.value(), self.k_max_input.value()]
        alpha_range = [self.alpha_min_input.value(), self.alpha_max_input.value()]
        metric = self.opt_metric_combo.currentText()
        selection_params = {k: v.value() for k, v in self.mode_inputs.items()}
        selection_params.update({'selection_method': self.selection_method_combo.currentText(), 'correlation_reference': self.corr_ref_combo.currentText()})
        
        fft_size = int(self.fft_size_combo.currentText())
        
        self.opt_progress.setValue(0)
        self.opt_progress.setVisible(True)
        self.opt_result_label.setText("Optimizing...")
        
        self.opt_thread = OptimizationThread(signal, K_range, alpha_range, metric, selection_params, fps, self.vmd_inputs['tau'].value(), self.vmd_inputs['dc'].value(), self.vmd_inputs['init'].value(), self.vmd_inputs['tol'].value(), fft_size)
        self.opt_thread.progress.connect(self.opt_progress.setValue)
        self.opt_thread.finished.connect(lambda K, a, s: self.on_optimization_complete(K, a, s))
        self.auto_opt_btn.setEnabled(False)
        self.opt_thread.start()

    def on_optimization_complete(self, best_K, best_alpha, best_score):
        self.opt_progress.setVisible(False)
        self.auto_opt_btn.setEnabled(True)
        self.vmd_inputs['k'].setValue(best_K)
        self.vmd_inputs['alpha'].setValue(best_alpha)
        
        metric_name = self.opt_metric_combo.currentText()
        self.opt_result_label.setText(f"Best: K={best_K}, α={best_alpha:.1f} ({metric_name}={best_score:.2f})")

    def display_vmd_results(self):
        self.clear_vmd_results()
        self.shared_x_axis = None
        self.shared_psd_axis = None
        if self.vmd_results is None: return
        
        # self.vmd_chart_rows is cleared in clear_vmd_results()

        fps = self.config['fps']
        original_signal = self.vmd_results['original']

        def add_row_vmd(title, data, color, hr=None, snr=None):
            canvas = self.add_chart_row(title, data, color, original_signal, hr, snr,
                                        target_layout=self.vmd_results_layout,
                                        target_chart_rows_list=self.vmd_chart_rows)
        
        # Calculate metrics on the true original signal
        hr_orig = computation.estimate_heart_rate(original_signal, fps)
        snr_orig = computation.calculate_snr(original_signal, original_signal, fps)
        
        # Create a zero-mean version for display only
        display_signal = original_signal - np.mean(original_signal)
        add_row_vmd("Raw Signal (Zero-Mean)", display_signal, "#3b82f6", hr_orig, snr_orig)
        
        if self.vmd_results['preprocessing_label'] != "None":
            prep = self.vmd_results['preprocessed']
            hr_prep = computation.estimate_heart_rate(prep, fps)
            snr_prep = computation.calculate_snr(original_signal, prep, fps)
            label = f"Preprocessed Signal ({self.vmd_results['preprocessing_label']})" if self.vmd_results['preprocessing_label'] else "Preprocessed Signal"
            add_row_vmd(label, prep, "#f59e0b", hr_prep, snr_prep)
            
        if self.show_imfs_btn.isChecked():
            for i, mode in enumerate(self.vmd_results['modes']):
                info = self.vmd_results['mode_info'][i]
                label = f"{'✓' if info['selected'] else '✗'} Mode {i+1} - {info['center_freq']:.2f} Hz | E: {info['energy']:.1f}% | C: {info['correlation']:.2f}"
                hr_mode = computation.estimate_heart_rate(mode, fps)
                snr_mode = computation.calculate_snr(original_signal, mode, fps)
                add_row_vmd(label, mode, "#10b981" if info['selected'] else "#9ca3af", hr_mode, snr_mode)
                
        recon = self.vmd_results['reconstructed']
        hr_recon = computation.estimate_heart_rate(recon, fps)
        snr_recon = computation.calculate_snr(original_signal, recon, fps)
        add_row_vmd("Reconstructed (All Modes)", recon, "#8b5cf6", hr_recon, snr_recon)
        
        add_row_vmd("Extracted rPPG (Selected)", self.vmd_results['extracted'], "#10b981", self.vmd_results['hr'], self.vmd_results['snr'])

        self.finalize_chart_layout(self.vmd_chart_rows)

    def run_all_filters(self):
        signal = self.get_selected_signal_segment()
        if signal is None or len(signal) == 0:
            QMessageBox.warning(self, "No Signal Selected", "Please select a CSV file from the list to analyze.")
            return
        self.run_filters_btn.setEnabled(False)
        self.run_filters_btn.setText("Processing...")
        QApplication.processEvents()
        
        # Update config from UI before running to ensure latest values are used and saved
        self.update_config_from_ui()
        
        params = {
            'signal': signal,
            'fps': self.config['fps'],
            'detrend': {'enabled': self.detrend_check.isChecked(), 'lambda': self.detrend_lambda.value()},
            'normalize': {'enabled': True, 'method': 'z-score'}, # Force enabled
            'harmonics': {
                'enabled': self.harmonics_check.isChecked(),
                'gain': self.harmonics_gain.value(),
                'freq_min': self.mode_inputs['freq_min'].value(),
                'freq_max': self.mode_inputs['freq_max'].value()
            },
            'filter_harmonics': {
                'enabled': self.filter_harmonics_check.isChecked(),
                'gain': self.filter_harmonics_gain.value(),
                'freq_min': self.mode_inputs['freq_min'].value(),
                'freq_max': self.mode_inputs['freq_max'].value()
            },
            'butterworth': {
                'enabled': self.butter_check.isChecked(),
                'order': self.butter_inputs['order'].value(),
                'freq_min': self.butter_inputs['freq_min'].value(),
                'freq_max': self.butter_inputs['freq_max'].value()
            },
            'chebyshev': {
                'enabled': self.cheby_check.isChecked(),
                'order': self.cheby_inputs['order'].value(),
                'ripple': self.cheby_inputs['ripple'].value(),
                'freq_min': self.cheby_inputs['freq_min'].value(),
                'freq_max': self.cheby_inputs['freq_max'].value()
            },
            'cheby2': {
                'enabled': self.cheby2_check.isChecked(),
                'order': self.cheby2_inputs['order'].value(),
                'stopband_atten': self.cheby2_inputs['stopband_atten'].value(),
                'freq_min': self.cheby2_inputs['freq_min'].value(),
                'freq_max': self.cheby2_inputs['freq_max'].value()
            },
            'elliptic': {
                'enabled': self.elliptic_check.isChecked(),
                'order': self.elliptic_inputs['order'].value(),
                'passband_ripple': self.elliptic_inputs['passband_ripple'].value(),
                'stopband_atten': self.elliptic_inputs['stopband_atten'].value(),
                'freq_min': self.elliptic_inputs['freq_min'].value(),
                'freq_max': self.elliptic_inputs['freq_max'].value()
            },
            'moving_average': {
                'enabled': self.ma_check.isChecked(),
                'window_size': self.ma_inputs['window_size'].value()
            },
            'savgol': {
                'enabled': self.savgol_check.isChecked(),
                'window_size': self.savgol_inputs['window_size'].value(),
                'poly_order': self.savgol_inputs['poly_order'].value()
            },
            'wavelet': {
                'enabled': self.wavelet_check.isChecked(),
                'wavelet': self.wavelet_inputs['wavelet'].currentText(),
                'level': self.wavelet_inputs['level'].value(),
                'threshold_mode': self.wavelet_inputs['threshold_mode'].currentText()
            }
        }
        
        self.filters_thread = ComputationThread(run_filters_computation, params)
        self.filters_thread.finished.connect(self.on_filters_finished)
        self.filters_thread.error.connect(self.on_computation_error)
        self.filters_thread.start()

    def on_filters_finished(self, results):
        self.filter_results = results
        self.display_filter_comparison()
        self.run_filters_btn.setEnabled(True)
        self.run_filters_btn.setText("▶ Run Selected Filters")

    def display_filter_comparison(self):
        self.clear_filter_results()
        self.shared_x_axis = None
        self.shared_psd_axis = None
        
        # self.filter_chart_rows is cleared in clear_filter_results()

        raw_signal = self.get_selected_signal_segment()
        if raw_signal is None: return
        
        # Apply zero-mean by default for display
        raw_signal_display = raw_signal - np.mean(raw_signal)
        
        fps = self.config['fps']
        hr_raw = computation.estimate_heart_rate(raw_signal, fps)
        snr_raw = computation.calculate_snr(raw_signal, raw_signal, fps)
        
        canvas = self.add_chart_row("Raw Signal (Zero-Mean)", raw_signal_display, "#64748b", raw_signal, hr_raw, snr_raw,
                                    target_layout=self.filter_results_layout,
                                    target_chart_rows_list=self.filter_chart_rows)
        
        if self.filter_results and 'preprocessed_signal' in self.filter_results:
            preproc_sig = self.filter_results['preprocessed_signal']
            hr_preproc = computation.estimate_heart_rate(preproc_sig, fps)
            snr_preproc = computation.calculate_snr(raw_signal, preproc_sig, fps)
            canvas = self.add_chart_row(f"Preprocessed ({self.filter_results['preproc_label']})", preproc_sig, "#f59e0b", raw_signal, hr_preproc, snr_preproc,
                                        target_layout=self.filter_results_layout,
                                        target_chart_rows_list=self.filter_chart_rows)

        if self.vmd_results:
            canvas = self.add_chart_row("VMD Extracted", self.vmd_results['extracted'], "#10b981", raw_signal, self.vmd_results['hr'], self.vmd_results['snr'],
                                        target_layout=self.filter_results_layout,
                                        target_chart_rows_list=self.filter_chart_rows)
            
        color_map = {
            'Harmonics Enhanced': '#ec4899', 'Butterworth': '#3b82f6', 'Chebyshev': '#ef4444',
            'Chebyshev II': '#f97316', 'Elliptic': '#84cc16',
            'Moving Average': '#f59e0b', 'Savitzky-Golay': '#8b5cf6', 'Wavelet Denoising': '#0ea5e9'
        }
        
        if hasattr(self, 'filter_results'):
            for name, data in self.filter_results.items():
                if name in ['preprocessed_signal', 'preproc_label']: continue
                color = color_map.get(name, '#06b6d4')
                canvas = self.add_chart_row(name, data['signal'], color, raw_signal, data['hr'], snr_raw,
                                            target_layout=self.filter_results_layout,
                                            target_chart_rows_list=self.filter_chart_rows)

        self.finalize_chart_layout(self.filter_chart_rows)

    def finalize_chart_layout(self, chart_rows_list):
        """Adjusts labels and spacing for all generated chart rows."""
        for i, canvas in enumerate(chart_rows_list):
            is_last_row = (i == len(chart_rows_list) - 1)
            
            # Configure X-axis labels
            canvas.ax_sig.tick_params(labelbottom=is_last_row)
            canvas.ax_spec.tick_params(labelbottom=is_last_row)
            canvas.ax_psd.tick_params(labelbottom=is_last_row)
            
            if is_last_row:
                canvas.ax_sig.set_xlabel("Time (s)")
                canvas.ax_spec.set_xlabel("Time (s)")
                canvas.ax_psd.set_xlabel("Frequency (Hz)")
                # Margins for row WITH labels
                canvas.figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.20, wspace=0.15)
            else:
                canvas.ax_sig.set_xlabel("")
                canvas.ax_spec.set_xlabel("")
                canvas.ax_psd.set_xlabel("")
                # Tighter margins for row WITHOUT labels
                canvas.figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.15)

            canvas.draw()

    def add_chart_row(self, title, data, color, original_signal, hr=None, snr=None, target_layout=None, target_chart_rows_list=None):
        """Adds a row with Signal, Spectrogram, and PSD charts."""
        height = self.config.get('chart_height', 200)
        fps = self.config['fps']
        fft_size = int(self.fft_size_combo.currentText())
        
        # Create optimized canvas with 3 subplots
        canvas = RowCanvas(title=title, width=14, height=3) # Pass title to RowCanvas
        canvas.setMinimumHeight(height)
        
        # --- Signal Chart (Left) ---
        if self.shared_x_axis:
            canvas.ax_sig.sharex(self.shared_x_axis)
        else:
            self.shared_x_axis = canvas.ax_sig
            
        time_axis = np.arange(len(data)) / fps
        canvas.ax_sig.plot(time_axis, data, color=color, linewidth=1.5, label=title)
        
        canvas.ax_sig.legend(loc='upper right', fontsize='x-small', framealpha=0.7)
        
        # Signal-specific metrics (Time Lag)
        signal_metrics_parts = []
        # Only calculate lag if the data is not the original signal itself
        if original_signal is not None and np.any(data != original_signal):
            time_lag = computation.calculate_time_lag(original_signal, data, fps)
            signal_metrics_parts.append(f"Lag: {time_lag:.2f} ms")

        if signal_metrics_parts:
            signal_metrics_text = " | ".join(signal_metrics_parts)
            canvas.ax_sig.text(0.02, 0.95, signal_metrics_text, transform=canvas.ax_sig.transAxes,
                                    verticalalignment='top', fontsize='small', fontweight='bold', color='#1e40af',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#e5e7eb'))
        
        canvas.ax_sig.grid(True, alpha=0.3)
        canvas.ax_sig.set_ylabel("Amp")
        
        def format_unit(x, pos):
            return f'{x*1e-3:.1f}k' if abs(x) >= 1000 else f'{x:.1f}'
        canvas.ax_sig.yaxis.set_major_formatter(FuncFormatter(format_unit))
        
        # --- Spectrogram Chart (Middle) ---
        nperseg = min(len(data)//2, fft_size)
        noverlap = min(len(data)//4, nperseg//2)
        f_spec, t_spec, Sxx = spectrogram(data, fs=fps, nperseg=nperseg, noverlap=noverlap)
        
        mask = f_spec <= 5
        if np.any(mask):
            f_spec_lim, Sxx_lim = f_spec[mask], Sxx[mask, :]
            
            canvas.ax_spec.imshow(
                10 * np.log10(Sxx_lim + 1e-10),
                extent=[time_axis[0], time_axis[-1], f_spec_lim[0], f_spec_lim[-1]],
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
        
        canvas.ax_spec.set_ylabel("Freq (Hz)")
        canvas.ax_spec.sharex(canvas.ax_sig)
        
        # --- PSD Chart (Right) ---
        nperseg_welch = min(fft_size, len(data))
        f, Pxx = welch(data, fs=fps, window='hann', nperseg=nperseg_welch)
        
        if self.shared_psd_axis:
            canvas.ax_psd.sharex(self.shared_psd_axis)
        else:
            self.shared_psd_axis = canvas.ax_psd
            
        canvas.ax_psd.plot(f, Pxx, color=color, linewidth=1) 
        canvas.ax_psd.fill_between(f, Pxx, color=color, alpha=0.2)

        # PSD-specific metrics (HR and SNR)
        psd_metrics_parts = []
        if hr is not None: psd_metrics_parts.append(f"Main Freq: {hr/60:.2f} Hz")
        if snr is not None: psd_metrics_parts.append(f"SNR: {snr:.1f} dB")

        if psd_metrics_parts:
            psd_metrics_text = "\n".join(psd_metrics_parts) # Join with newline for vertical stacking
            canvas.ax_psd.text(0.98, 0.95, psd_metrics_text, transform=canvas.ax_psd.transAxes,
                                    verticalalignment='top', horizontalalignment='right', # Align to top-right
                                    fontsize='small', fontweight='bold', color='#1e40af',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#e5e7eb'))

        canvas.ax_psd.set_xlim(0, 5)
        canvas.ax_psd.tick_params(axis='both', which='major', labelsize='x-small')
        canvas.ax_psd.yaxis.set_major_formatter(FuncFormatter(format_unit))
        canvas.ax_psd.grid(True, alpha=0.3)
        
        # Add to the specified layout and list
        if target_layout:
            target_layout.addWidget(canvas)
        if target_chart_rows_list is not None:
            target_chart_rows_list.append(canvas)
            
        return canvas

    def update_ui_from_config(self):
        """Restores UI state from self.config"""
        # 1. Subject/Signal Info
        start = self.config.get('signal', {}).get('start_time', 1.0)
        duration = self.config.get('signal', {}).get('duration', 10.0)
        fps = self.config.get('fps', 30)
        chart_height = self.config.get('chart_height', 200)
        
        for controls in self.shared_controls:
            controls['start'].blockSignals(True)
            controls['dur'].blockSignals(True)
            controls['fps'].blockSignals(True)
            controls['chart_height'].blockSignals(True)
            
            controls['start'].setValue(start)
            controls['dur'].setValue(duration)
            controls['fps'].setValue(fps)
            controls['chart_height'].setValue(chart_height)
            
            controls['start'].blockSignals(False)
            controls['dur'].blockSignals(False)
            controls['fps'].blockSignals(False)
            controls['chart_height'].blockSignals(False)

        # 2. Preprocessing
        pre = self.config.get('preprocessing', {})
        self.detrend_check.setChecked(pre.get('detrend', {}).get('enabled', True))
        self.detrend_lambda.setValue(pre.get('detrend', {}).get('lambda', 100))
        self.harmonics_check.setChecked(pre.get('harmonics', {}).get('enabled', False))
        self.harmonics_gain.setValue(pre.get('harmonics', {}).get('harmonics_gain', 2.0))
        self.fft_size_combo.setCurrentText(str(pre.get('fft_size', 1024)))

        # 3. VMD Params
        vmd = self.config.get('vmd', {})
        for k, v in self.vmd_inputs.items():
            if k.upper() in vmd:
                v.setValue(vmd[k.upper()])

        # 4. Mode Selection
        ms = self.config.get('mode_selection', {})
        for k, v in self.mode_inputs.items():
            if k in ms:
                v.setValue(ms[k])
        self.selection_method_combo.setCurrentText(ms.get('selection_method', 'all_criteria'))
        self.corr_ref_combo.setCurrentText(ms.get('correlation_reference', 'original'))

        # 5. Optimization
        opt = self.config.get('auto_optimize', {})
        if 'K_range' in opt:
            self.k_min_input.setValue(opt['K_range'][0])
            self.k_max_input.setValue(opt['K_range'][1])
        if 'alpha_range' in opt:
            self.alpha_min_input.setValue(opt['alpha_range'][0])
            self.alpha_max_input.setValue(opt['alpha_range'][1])
        self.opt_metric_combo.setCurrentText(opt.get('optimization_metric', 'snr'))

        # 6. Traditional Filters
        tf = self.config.get('traditional_filters', {})
        
        # Helper to set filter group
        def set_filter(group, inputs, key):
            data = tf.get(key, {})
            group.setChecked(data.get('enabled', True)) # Default to True
            for k, input_widget in inputs.items():
                if k in data:
                    if isinstance(input_widget, (QSpinBox, QDoubleSpinBox)):
                        input_widget.setValue(data[k])
                    elif isinstance(input_widget, QComboBox):
                        input_widget.setCurrentText(data[k])

        set_filter(self.butter_check, self.butter_inputs, 'butterworth')
        set_filter(self.cheby_check, self.cheby_inputs, 'chebyshev')
        set_filter(self.cheby2_check, self.cheby2_inputs, 'cheby2')
        set_filter(self.elliptic_check, self.elliptic_inputs, 'elliptic')
        set_filter(self.ma_check, self.ma_inputs, 'moving_average')
        set_filter(self.savgol_check, self.savgol_inputs, 'savgol')
        set_filter(self.wavelet_check, self.wavelet_inputs, 'wavelet')
        
        if 'harmonics' in tf:
            self.filter_harmonics_check.setChecked(tf['harmonics'].get('enabled', False))
            self.filter_harmonics_gain.setValue(tf['harmonics'].get('harmonics_gain', 2.0))

        # 7. Pipeline Steps
        # Clear existing steps
        while len(self.pipeline_steps) > 0:
            step = self.pipeline_steps[0]
            self.remove_pipeline_step(step['widget'])
            
        # Reconstruct from config
        pipeline_data = self.config.get('pipeline_steps', [])
        for step_info in pipeline_data:
            self.filter_type_combo.setCurrentText(step_info['type'])
            self.add_pipeline_step()
            # Set params for the last added step
            step = self.pipeline_steps[-1]
            for param_name, param_value in step_info.get('params', {}).items():
                if param_name in step['params']:
                    input_widget = step['params'][param_name]
                    if isinstance(input_widget, (QSpinBox, QDoubleSpinBox)):
                        input_widget.setValue(param_value)
                    elif isinstance(input_widget, QComboBox):
                        input_widget.setCurrentText(param_value)

        self.update_chart_heights()

    def update_config_from_ui(self):
        # Preprocessing
        self.config['preprocessing']['detrend'] = {
            'enabled': self.detrend_check.isChecked(),
            'lambda': self.detrend_lambda.value()
        }
        self.config['preprocessing']['normalization'] = {
            'enabled': True, # Always enabled
            'method': 'z-score'
        }
        self.config['preprocessing']['harmonics'] = {
            'enabled': self.harmonics_check.isChecked(),
            'harmonics_gain': self.harmonics_gain.value(),
            'freq_min': self.mode_inputs['freq_min'].value(),
            'freq_max': self.mode_inputs['freq_max'].value()
        }
        self.config['preprocessing']['fft_size'] = int(self.fft_size_combo.currentText())
        
        # VMD Params
        for k, v in self.vmd_inputs.items():
            self.config['vmd'][k.upper()] = v.value()
            
        # Mode Selection
        for k, v in self.mode_inputs.items():
            self.config['mode_selection'][k] = v.value()
        self.config['mode_selection']['selection_method'] = self.selection_method_combo.currentText()
        self.config['mode_selection']['correlation_reference'] = self.corr_ref_combo.currentText()
        
        # Optimization
        self.config['auto_optimize']['K_range'] = [self.k_min_input.value(), self.k_max_input.value()]
        self.config['auto_optimize']['alpha_range'] = [self.alpha_min_input.value(), self.alpha_max_input.value()]
        self.config['auto_optimize']['optimization_metric'] = self.opt_metric_combo.currentText()
        
        # UI State
        self.config['ui_state']['show_individual_modes'] = self.show_imfs_btn.isChecked()
        
        # Traditional Filters
        self.config['traditional_filters']['butterworth'] = {
            'enabled': self.butter_check.isChecked(),
            'order': self.butter_inputs['order'].value(),
            'freq_min': self.butter_inputs['freq_min'].value(),
            'freq_max': self.butter_inputs['freq_max'].value()
        }
        
        self.config['traditional_filters']['chebyshev'] = {
            'enabled': self.cheby_check.isChecked(),
            'order': self.cheby_inputs['order'].value(),
            'ripple': self.cheby_inputs['ripple'].value(),
            'freq_min': self.cheby_inputs['freq_min'].value(),
            'freq_max': self.cheby_inputs['freq_max'].value()
        }
        
        self.config['traditional_filters']['cheby2'] = {
            'enabled': self.cheby2_check.isChecked(),
            'order': self.cheby2_inputs['order'].value(),
            'stopband_atten': self.cheby2_inputs['stopband_atten'].value(),
            'freq_min': self.cheby2_inputs['freq_min'].value(),
            'freq_max': self.cheby2_inputs['freq_max'].value()
        }

        self.config['traditional_filters']['elliptic'] = {
            'enabled': self.elliptic_check.isChecked(),
            'order': self.elliptic_inputs['order'].value(),
            'passband_ripple': self.elliptic_inputs['passband_ripple'].value(),
            'stopband_atten': self.elliptic_inputs['stopband_atten'].value(),
            'freq_min': self.elliptic_inputs['freq_min'].value(),
            'freq_max': self.elliptic_inputs['freq_max'].value()
        }
        
        self.config['traditional_filters']['moving_average'] = {
            'enabled': self.ma_check.isChecked(),
            'window_size': self.ma_inputs['window_size'].value()
        }
        
        self.config['traditional_filters']['savgol'] = {
            'enabled': self.savgol_check.isChecked(),
            'window_size': self.savgol_inputs['window_size'].value(),
            'poly_order': self.savgol_inputs['poly_order'].value()
        }
        
        self.config['traditional_filters']['wavelet'] = {
            'enabled': self.wavelet_check.isChecked(),
            'wavelet': self.wavelet_inputs['wavelet'].currentText(),
            'level': self.wavelet_inputs['level'].value(),
            'threshold_mode': self.wavelet_inputs['threshold_mode'].currentText()
        }
        
        self.config['traditional_filters']['harmonics'] = {
            'enabled': self.filter_harmonics_check.isChecked(),
            'harmonics_gain': self.filter_harmonics_gain.value()
        }
        
        # Shared Controls (Signal)
        if self.shared_controls:
            controls = self.shared_controls[0]
            if 'signal' not in self.config: self.config['signal'] = {}
            self.config['signal']['start_time'] = controls['start'].value()
            self.config['signal']['duration'] = controls['dur'].value()
            self.config['fps'] = controls['fps'].value()
            self.config['chart_height'] = controls['chart_height'].value()
            
        # Pipeline Steps
        pipeline_data = []
        for step in self.pipeline_steps:
            step_params = {}
            for name, input_widget in step['params'].items():
                if isinstance(input_widget, (QSpinBox, QDoubleSpinBox)):
                    step_params[name] = input_widget.value()
                elif isinstance(input_widget, QComboBox):
                    step_params[name] = input_widget.currentText()
            pipeline_data.append({
                'type': step['type'],
                'params': step_params
            })
        self.config['pipeline_steps'] = pipeline_data
            
        self.update_chart_heights()

    def open_save_data_dialog(self):
        # Get the current project root (assuming the script is run from within the project)
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_save_path = os.path.join(project_root, self.config.get('save_path', 'saved_data'))

        dialog = SaveDataDialog(
            self,
            vmd_available=self.vmd_results is not None,
            filters_available=self.filter_results is not None,
            initial_save_path=default_save_path
        )
        if dialog.exec():
            options = dialog.get_save_options()
            
            # Update config if save path changed
            new_base_path = options['base_path']
            if new_base_path != default_save_path:
                # Extract relative path if possible, otherwise save absolute
                try:
                    relative_path = os.path.relpath(new_base_path, project_root)
                    self.config['save_path'] = relative_path
                except ValueError: # If on different drive
                    self.config['save_path'] = new_base_path
                config.save_config(self.config) # Save updated config

            self.execute_save_data(options)

    def execute_save_data(self, options):
        base_path = options['base_path']
        folder_name = options['folder_name']

        if not base_path:
            QMessageBox.warning(self, "Path Required", "Please select a base directory to save the data.")
            return

        output_path = os.path.join(base_path, folder_name) if folder_name else base_path
        
        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Error Creating Directory", f"Could not create directory:\n{output_path}\n\nError: {e}")
            return

        # Save Settings
        if options['save_settings']:
            self.update_config_from_ui()
            config.save_config(self.config, os.path.join(output_path, "settings.json"))

        # Save Figures
        if options['save_figure']:
            filename = os.path.basename(self.selected_file_path) if self.selected_file_path else "rPPG_Analysis"
            if self.vmd_results:
                vmd_plots_dir = os.path.join(output_path, "vmd_plots")
                os.makedirs(vmd_plots_dir, exist_ok=True)
                self.save_stitched_figure(self.vmd_chart_rows, os.path.join(vmd_plots_dir, "vmd_all_plots.png"), global_title=f"VMD Analysis: {filename}")
                # Also save individual plots for VMD
                for canvas in self.vmd_chart_rows:
                    canvas.figure.suptitle(f"{canvas.title} - {filename}", fontsize=9)
                    orig_top = canvas.figure.subplotpars.top
                    canvas.figure.subplots_adjust(top=0.88) # Tighter margin
                    filename_safe = "".join([c for c in canvas.title if c.isalnum() or c in (' ', '_')]).rstrip()
                    filename_safe = filename_safe.replace(' ', '_').lower()
                    canvas.figure.savefig(os.path.join(vmd_plots_dir, f"{filename_safe}.png"), bbox_inches='tight')
                    canvas.figure.suptitle("") # Clear title after saving
                    canvas.figure.subplots_adjust(top=orig_top) # Restore margin

            if self.filter_results:
                filter_plots_dir = os.path.join(output_path, "filter_plots")
                os.makedirs(filter_plots_dir, exist_ok=True)
                self.save_stitched_figure(self.filter_chart_rows, os.path.join(filter_plots_dir, "filter_all_plots.png"), global_title=f"Filter Comparison: {filename}")
                # Also save individual plots for Filters
                for canvas in self.filter_chart_rows:
                    canvas.figure.suptitle(f"{canvas.title} - {filename}", fontsize=9)
                    orig_top = canvas.figure.subplotpars.top
                    canvas.figure.subplots_adjust(top=0.88) # Tighter margin
                    filename_safe = "".join([c for c in canvas.title if c.isalnum() or c in (' ', '_')]).rstrip()
                    filename_safe = filename_safe.replace(' ', '_').lower()
                    canvas.figure.savefig(os.path.join(filter_plots_dir, f"{filename_safe}.png"), bbox_inches='tight')
                    canvas.figure.suptitle("") # Clear title after saving
                    canvas.figure.subplots_adjust(top=orig_top) # Restore margin

            if self.pipeline_chart_rows:
                pipeline_plots_dir = os.path.join(output_path, "pipeline_plots")
                os.makedirs(pipeline_plots_dir, exist_ok=True)
                self.save_stitched_figure(self.pipeline_chart_rows, os.path.join(pipeline_plots_dir, "pipeline_all_plots.png"), global_title=f"Filter Pipeline: {filename}")
                for canvas in self.pipeline_chart_rows:
                    canvas.figure.suptitle(f"{canvas.title} - {filename}", fontsize=9)
                    orig_top = canvas.figure.subplotpars.top
                    canvas.figure.subplots_adjust(top=0.88) # Tighter margin
                    filename_safe = "".join([c for c in canvas.title if c.isalnum() or c in (' ', '_')]).rstrip()
                    filename_safe = filename_safe.replace(' ', '_').lower()
                    canvas.figure.savefig(os.path.join(pipeline_plots_dir, f"{filename_safe}.png"), bbox_inches='tight')
                    canvas.figure.suptitle("") # Clear title after saving
                    canvas.figure.subplots_adjust(top=orig_top) # Restore margin

        # Save VMD Analysis Signals
        if options['save_vmd'] and self.vmd_results:
            data_to_save = {
                'raw_signal': self.vmd_results['original'],
                'preprocessed_signal': self.vmd_results.get('preprocessed'),
                'reconstructed_signal': self.vmd_results.get('reconstructed'),
                'extracted_signal': self.vmd_results.get('extracted')
            }
            df_dict = {k: pd.Series(v) for k, v in data_to_save.items() if v is not None}
            df = pd.DataFrame(df_dict)
            df.to_csv(os.path.join(output_path, "vmd_analysis_signals.csv"), index=False)

        # Save IMFs
        if options['save_imfs'] and self.vmd_results:
            imf_dict = {f'imf_{i+1}': mode for i, mode in enumerate(self.vmd_results['modes'])}
            df = pd.DataFrame(imf_dict)
            df.to_csv(os.path.join(output_path, "imfs.csv"), index=False)

        # Save Filtered Signals
        if options['save_filters'] and self.filter_results:
            raw_signal_segment = self.get_selected_signal_segment() # Get the raw segment used for this run
            signals_to_save = {'raw_signal': raw_signal_segment}
            
            if 'preprocessed_signal' in self.filter_results:
                signals_to_save['preprocessed_signal'] = self.filter_results['preprocessed_signal']
            
            for name, data in self.filter_results.items():
                if name not in ['preprocessed_signal', 'preproc_label']:
                    signals_to_save[name.lower().replace(' ', '_')] = data['signal']
            
            df = pd.DataFrame({k: pd.Series(v) for k, v in signals_to_save.items()})
            df.to_csv(os.path.join(output_path, "filtered_signals.csv"), index=False)

        # Save Pipeline Signals
        if options.get('save_pipeline') and self.pipeline_chart_rows:
            pipeline_signals = {}
            for canvas in self.pipeline_chart_rows:
                # Extract data from the plot
                line = canvas.ax_sig.get_lines()[0]
                pipeline_signals[canvas.title.lower().replace(' ', '_')] = line.get_ydata()
            
            df = pd.DataFrame({k: pd.Series(v) for k, v in pipeline_signals.items()})
            df.to_csv(os.path.join(output_path, "pipeline_signals.csv"), index=False)
        
        QMessageBox.information(self, "Save Complete", f"Data successfully saved to:\n{output_path}")

    def save_stitched_figure(self, chart_rows, output_path, global_title=None):
        if not chart_rows:
            return

        images = []
        for canvas in chart_rows:
            # Ensure the canvas is drawn before buffering its content
            canvas.draw() 
            # Convert memoryview to bytes
            buf = bytes(canvas.buffer_rgba())
            img = Image.frombytes("RGBA", (canvas.width(), canvas.height()), buf)
            images.append(img)

        if not images:
            return

        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        
        title_height = 0
        if global_title:
            title_height = 40 # Reduced space for smaller font
            total_height += title_height

        stitched_image = Image.new('RGBA', (max_width, total_height), (255, 255, 255, 255))
        
        y_offset = 0
        if global_title:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(stitched_image)
            # Try to get a nice font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 20) # Reduced font size
            except:
                font = ImageFont.load_default()
            
            # Draw centered title
            text_width = draw.textlength(global_title, font=font) if hasattr(draw, 'textlength') else 200
            draw.text(((max_width - text_width) // 2, 8), global_title, fill=(0, 0, 0, 255), font=font)
            y_offset = title_height

        for img in images:
            stitched_image.paste(img, (0, y_offset))
            y_offset += img.height

        stitched_image.save(output_path)

    def load_default_config(self):
        default_path = config.DEFAULT_CONFIG_FILE
        if os.path.exists(default_path):
            self.config = config.load_config(default_path)
            self.update_ui_from_config()
            QMessageBox.information(self, "Config Loaded", "Default configuration loaded.")
        else:
            QMessageBox.warning(self, "Error", "Default config file not found.")

    def load_custom_config(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", config.CONFIG_DIR, "JSON Files (*.json)")
        if filepath:
            try:
                self.config = config.load_config(filepath)
                self.update_ui_from_config()
                QMessageBox.information(self, "Config Loaded", f"Configuration loaded from {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    def save_current_config(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", config.CONFIG_DIR, "JSON Files (*.json)")
        if filepath:
            self.update_config_from_ui()
            if config.save_config(self.config, filepath):
                QMessageBox.information(self, "Config Saved", f"Configuration saved to {filepath}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save configuration.")

    def closeEvent(self, event):
        self.update_config_from_ui()
        config.save_config(self.config)
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = VMDrPPGMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
