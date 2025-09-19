# App Configuration
APP_TITLE = 'FACS Data Analysis'
PAGE_TITLE = 'FACSデータ解析'
PAGE_ICON = '🔬'
LAYOUT = 'wide'

# File Upload Configuration
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = ['.fcs']

# Data Processing Configuration
MIN_EVENTS = 1000
MAX_EVENTS = 100000
DEFAULT_MAX_EVENTS = 50000

# Plot Configuration
DEFAULT_BINS = 100
MAX_BINS = 500
DEFAULT_ALPHA = 0.7
COLORMAP_OPTIONS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
DEFAULT_COLORMAP = 'viridis'
PLOT_TYPES = ['scatter', 'density', 'histogram']

# Gate Configuration
GATE_TYPES = {
    'threshold': '閾値ゲート',
    'rectangular': '矩形ゲート',
    'polygon': 'ポリゴンゲート',
    'ellipse': '楕円ゲート'
}
GATE_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
DEFAULT_GATE_COLOR = 'red'
MAX_GATES = 10

# Data Transform Methods
TRANSFORM_METHODS = {
    'none': 'なし',
    'log10': 'Log10',
    'asinh': 'Asinh'
}

# Main Tab Configuration
MAIN_TABS = ['📊 基本情報', '📈 可視化', '🎯 ゲーティング', '📋 統計解析']

# Common Channel Mappings
CHANNEL_MAPPINGS = {
    'FSC-A': 'Forward Scatter Area',
    'FSC-H': 'Forward Scatter Height',
    'SSC-A': 'Side Scatter Area',
    'SSC-H': 'Side Scatter Height',
    'FITC-A': 'FITC Area',
    'PE-A': 'PE Area',
    'APC-A': 'APC Area',
    'PerCP-A': 'PerCP Area'
}

# Statistics Configuration
BASIC_STATISTICS = ['count', 'mean', 'median', 'std', 'min', 'max']
PERCENTILES = [25, 50, 75, 95]

# Export Configuration
CSV_SEPARATOR = ','
FILE_PREFIX = 'facs_analysis_'
