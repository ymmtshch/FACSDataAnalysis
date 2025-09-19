"""
Multilingual message definitions for FACS Data Analysis App
Supports Japanese (ja) and English (en)
"""

MESSAGES = {
    "ja": {
        "error": {
    "file_too_large": "ファイルサイズが100MBを超えています。",
    "invalid_file": "FCSファイルをアップロードしてください。",
    "file_read_error": "ファイルの読み込みに失敗しました。",
    "insufficient_data": "データが不十分です。",
    "memory_error": "メモリ不足です。データサイズを小さくしてください。",
    "gating_error": "ゲーティング処理中にエラーが発生しました。"
},
        "success": {
    "file_loaded": "ファイルが正常に読み込まれました。",
    "gate_created": "ゲートが正常に作成されました。",
    "data_exported": "データが正常にエクスポートされました。"
},
        "warning": {
    "large_file": "ファイルサイズが大きいため、処理に時間がかかる場合があります。",
    "many_events": "イベント数が多いため、表示をサブサンプリングしています。",
    "memory_optimization": "メモリ最適化のため、サンプリングが適用されています。"
}
    },
    "en": {
        "error": {
    "file_too_large": "File size exceeds 100MB.",
    "invalid_file": "Please upload an FCS file.",
    "file_read_error": "Failed to read the file.",
    "insufficient_data": "Insufficient data.",
    "memory_error": "Memory error. Please reduce the data size.",
    "gating_error": "An error occurred during gating."
},
        "success": {
    "file_loaded": "File loaded successfully.",
    "gate_created": "Gate created successfully.",
    "data_exported": "Data exported successfully."
},
        "warning": {
    "large_file": "Large file size may cause slow processing.",
    "many_events": "Subsampling applied due to large number of events.",
    "memory_optimization": "Sampling applied for memory optimization."
}
    }
}

def get_message(message_type, key, lang="ja"):
    """Get localized message"""
    return MESSAGES.get(lang, {}).get(message_type, {}).get(key, "")
