import json
import os
import hashlib

class User:
    def __init__(self, username, password_hash, role="operator"):
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def to_dict(self):
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role
        }

    @staticmethod
    def from_dict(data):
        return User(data["username"], data["password_hash"], data.get("role", "operator"))

class UserManager:
    def __init__(self, config_path="config/users.json"):
        self.config_path = config_path
        self.users = {}
        self.current_user = None
        self.load_users()

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def load_users(self):
        if not os.path.exists(self.config_path):
            # Create default admin user
            admin = User("admin", self._hash_password("123"), "admin")
            self.users = {"admin": admin}
            self.save_users()
        else:
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    for username, user_data in data.items():
                        self.users[username] = User.from_dict(user_data)
            except (json.JSONDecodeError, FileNotFoundError):
                # Fallback if file is corrupted
                admin = User("admin", self._hash_password("123"), "admin")
                self.users = {"admin": admin}
                self.save_users()

    def save_users(self):
        data = {username: user.to_dict() for username, user in self.users.items()}
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=4)

    def login(self, username, password):
        user = self.users.get(username)
        if user and user.password_hash == self._hash_password(password):
            self.current_user = user
            return True
        return False

    def logout(self):
        self.current_user = None

    def add_user(self, username, password, role="operator"):
        valid_roles = ["master", "tecnico", "operador"]
        if role not in valid_roles:
            raise ValueError(f"Invalid role. Must be one of {valid_roles}")
            
        if username in self.users:
            return False
            
        self.users[username] = User(username, self._hash_password(password), role)
        self.save_users()
        return True

    def delete_user(self, username):
        if username in self.users:
            if username == "admin":  # Prevent deleting the main admin
                return False
            del self.users[username]
            self.save_users()
            return True
        return False

    def get_all_users(self):
        return list(self.users.values())
