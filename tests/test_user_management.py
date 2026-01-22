import os
import sys
from src.auth import UserManager

def test_user_management():
    # Remove config file to start fresh
    if os.path.exists("config/users.json"):
        os.remove("config/users.json")
    
    um = UserManager()
    
    # Test adding valid roles
    assert um.add_user("operator1", "pass", "operador"), "Should add operator"
    assert um.add_user("tech1", "pass", "tecnico"), "Should add tecnico"
    assert um.add_user("admin2", "pass", "admin"), "Should add admin"
    
    # Test checking roles
    assert um.users["operator1"].role == "operador"
    assert um.users["tech1"].role == "tecnico"
    
    # Test adding invalid role
    try:
        um.add_user("baduser", "pass", "invalid_role")
        print("FAIL: Should not accept invalid_role")
        sys.exit(1)
    except ValueError:
        print("PASS: Rejected invalid_role")
        
    # Test deleting user
    assert um.delete_user("operator1"), "Should delete user"
    assert "operator1" not in um.users, "User should be gone"
    
    # Test delete admin protection
    assert not um.delete_user("admin"), "Should not delete main admin"
    
    # Check count
    assert len(um.get_all_users()) == 3 # admin, tech1, admin2
    
    print("All User Management tests passed.")

if __name__ == "__main__":
    test_user_management()
