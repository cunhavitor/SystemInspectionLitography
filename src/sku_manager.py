import json
import os
import shutil

class SKUManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.skus = {}
        self.active_sku = "default"
        self._load_config()

    def _load_config(self):
        """Load SKUs from JSON file."""
        if not os.path.exists(self.config_path):
            # Create default if not exists
            self.skus = {
                "default": {
                    "name": "Default SKU",
                    "model_path": "",
                    "reference_path": ""
                }
            }
            self.active_sku = "default"
            self.save_config()
        else:
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.skus = data.get("skus", {})
                    self.active_sku = data.get("active_sku", "default")
            except Exception as e:
                print(f"Error loading SKU config: {e}")
                self.skus = {}
                self.active_sku = "default"

    def save_config(self):
        """Save current SKUs to JSON file."""
        data = {
            "active_sku": self.active_sku,
            "skus": self.skus
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving SKU config: {e}")

    def get_active_sku(self):
        """Get the configuration dictionary for the active SKU."""
        return self.skus.get(self.active_sku, {})

    def get_sku_names(self):
        """Returns a list of display names for all SKUs."""
        return [sku["name"] for sku in self.skus.values()]

    def get_sku_by_name(self, name):
        """Find SKU key by its display name."""
        for key, data in self.skus.items():
            if data["name"] == name:
                return key, data
        return None, None

    def set_active_sku_by_name(self, name):
        """Set active SKU by its display name."""
        key, _ = self.get_sku_by_name(name)
        if key:
            self.active_sku = key
            self.save_config()
            return True
        return False

    def add_sku(self, name, model_path, reference_path):
        """Add a new SKU configuration."""
        # Generate a simple key (e.g., lowercase name with underscores)
        key = name.lower().replace(" ", "_")
        
        # Ensure unique key
        base_key = key
        counter = 1
        while key in self.skus:
            key = f"{base_key}_{counter}"
            counter += 1
            
        self.skus[key] = {
            "name": name,
            "model_path": model_path,
            "reference_path": reference_path
        }
        self.save_config()
        return key
