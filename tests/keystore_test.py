from remote_llm.keystore import ApiKeystore

def test_add_admin_key(tmp_path):
    keystore = ApiKeystore(f"sqlite:///{tmp_path}/keystore.db")
    keystore.add_admin_key(name="admin", key="482fdd5f-b59c-43de-98b9-4e19a21b4d85")
    assert keystore.get_key(name="admin") == "482fdd5f-b59c-43de-98b9-4e19a21b4d85"

def test_add_key(tmp_path):
    keystore = ApiKeystore(f"sqlite:///{tmp_path}/keystore.db")
    key = keystore.add_key(name="test")
    assert keystore.get_key(name="test") == key
    assert key != None

def test_check_key(tmp_path):
    keystore = ApiKeystore(f"sqlite:///{tmp_path}/keystore.db")
    key = keystore.add_key(name="test")
    assert keystore.check_key(key=key) == "test"
    assert keystore.check_key(key="invalid") == None