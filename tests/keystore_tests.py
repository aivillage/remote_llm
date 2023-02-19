from remote_llm.keystore import ApiKeystore

keystore = ApiKeystore("sqlite:///./keystore.db")

def test_add_admin_key():
    keystore.add_admin_key(name="admin", key="482fdd5f-b59c-43de-98b9-4e19a21b4d85")
    assert keystore.get_key(name="admin") == "482fdd5f-b59c-43de-98b9-4e19a21b4d85"

def test_add_key():
    key = keystore.add_key(name="test")
    assert keystore.get_key(name="test") == key
    assert key != None