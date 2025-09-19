# Utility Functions
def get_transform_methods():
    """Get available transform methods"""
    return TRANSFORM_METHODS

def get_gate_types():
    """Get available gate types"""
    return GATE_TYPES

def get_export_filename(base_filename, export_type='data'):
    """Generate export filename with simplified logic"""
    if not base_filename:
        base_filename = 'facs_data'
    
    # Remove extension if present
    if base_filename.endswith('.fcs'):
        base_filename = base_filename[:-4]
    
    if export_type == 'statistics' or export_type == 'stats':
        return f"{base_filename}_stats.csv"
    elif export_type == 'gate' or export_type == 'gate_data':
        return f"{base_filename}_gate_data.csv"
    else:
        return f"{base_filename}_data.csv"

def validate_max_events(num_events):
    """Validate maximum events setting with safe defaults"""
    if not isinstance(num_events, (int, float)) or num_events <= 0:
        return DEFAULT_MAX_EVENTS
    
    return max(MIN_EVENTS, min(int(num_events), MAX_EVENTS))

def validate_file_size(file_size_bytes):
    """Validate uploaded file size"""
    if not isinstance(file_size_bytes, (int, float)) or file_size_bytes <= 0:
        return False
    
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    return file_size_bytes <= max_size_bytes

def get_message(message_type, key):
    """Get localized message with fallback"""
    messages = {
        'error': ERROR_MESSAGES,
        'success': SUCCESS_MESSAGES,
        'warning': WARNING_MESSAGES
    }
    
    fallback_messages = {
        'error': 'エラーが発生しました。',
        'success': '処理が正常に完了しました。',
        'warning': '警告: データを確認してください。'
    }
    
    message_dict = messages.get(message_type, {})
    return message_dict.get(key, fallback_messages.get(message_type, ''))

def get_error_message(error_key):
    """Get localized error message"""
    return get_message('error', error_key)

def get_success_message(success_key):
    """Get localized success message"""
    return get_message('success', success_key)

def get_warning_message(warning_key):
    """Get localized warning message"""
    return get_message('warning', warning_key)

def is_valid_channel(channel_name):
    """Check if channel name is valid"""
    if not isinstance(channel_name, str) or not channel_name.strip():
        return False
    return True

def get_channel_display_name(channel_name):
    """Get display name for channel with fallback"""
    if not is_valid_channel(channel_name):
        return 'Unknown Channel'
    
    return CHANNEL_MAPPINGS.get(channel_name, channel_name)

def validate_bins(bins):
    """Validate histogram bins with safe defaults"""
    if not isinstance(bins, (int, float)) or bins <= 0:
        return DEFAULT_BINS
    
    return max(10, min(int(bins), MAX_BINS))

def validate_alpha(alpha):
    """Validate alpha value with safe defaults"""
    if not isinstance(alpha, (int, float)):
        return DEFAULT_ALPHA
    
    return max(0.1, min(float(alpha), 1.0))

def get_default_colormap():
    """Get default colormap"""
    return DEFAULT_COLORMAP

def is_valid_colormap(colormap):
    """Check if colormap is valid"""
    return colormap in COLORMAP_OPTIONS

def get_valid_colormap(colormap):
    """Get valid colormap with fallback"""
    if is_valid_colormap(colormap):
        return colormap
    return DEFAULT_COLORMAP

def get_gate_color(index=0):
    """Get gate color by index with fallback"""
    if not isinstance(index, int) or index < 0:
        return DEFAULT_GATE_COLOR
    
    if index < len(GATE_COLORS):
        return GATE_COLORS[index]
    
    return GATE_COLORS[index % len(GATE_COLORS)]

def can_add_gate(current_gates=0):
    """Check if more gates can be added"""
    if not isinstance(current_gates, int) or current_gates < 0:
        return True
    
    return current_gates < MAX_GATES

# Backward compatibility class (simplified)
class Config:
    """Simplified configuration class for backward compatibility"""
    
    # Basic app settings
    APP_TITLE = APP_TITLE
    PAGE_TITLE = PAGE_TITLE
    PAGE_ICON = PAGE_ICON
    
    # File settings
    MAX_FILE_SIZE = MAX_FILE_SIZE_MB
    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    
    # Plot settings
    DEFAULT_BINS = DEFAULT_BINS
    MAX_BINS = MAX_BINS
    DEFAULT_ALPHA = DEFAULT_ALPHA
    COLORMAP_OPTIONS = COLORMAP_OPTIONS
    DEFAULT_COLORMAP = DEFAULT_COLORMAP
    
    # Gate settings
    GATE_COLORS = GATE_COLORS
    DEFAULT_GATE_COLOR = DEFAULT_GATE_COLOR
    
    # Data settings
    MAX_EVENTS_DISPLAY = MAX_EVENTS
    CHANNEL_MAPPINGS = CHANNEL_MAPPINGS
    
    @classmethod
    def get_error_message(cls, error_key):
        return get_error_message(error_key)
    
    @classmethod
    def get_success_message(cls, success_key):
        return get_success_message(success_key)
    
    @classmethod
    def get_warning_message(cls, warning_key):
        return get_warning_message(warning_key)