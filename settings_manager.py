import json

SETTINGS_FILE = "user_settings.json"

def save_settings(settings_dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=4)
        print(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)
        print(f"Settings loaded from {SETTINGS_FILE}")
        return settings
    except FileNotFoundError:
        print("Settings file not found. Returning empty settings.")
        return {}
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}