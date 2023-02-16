
set OPENSSL_CONF=c:\OpenSSL-Win64\bin\openssl.cfg   

echo Generate CA key:
openssl genrsa -passout pass:aivillage -des3 -out server_keys/ca.key 4096

echo Generate CA certificate:
openssl req -passin pass:aivillage -new -x509 -days 365 -key server_keys/ca.key -out server_keys/ca.crt -subj  "/C=US/ST=MD/CN=MyRootCA"

echo Generate server key:
openssl genrsa -passout pass:aivillage -des3 -out server_keys/server.key 4096

echo Generate server signing request:
openssl req -passin pass:aivillage -new -key server_keys/server.key -out server_keys/server.csr -subj  "/C=US/ST=MD/CN=localhost"

echo Self-sign server certificate:
openssl x509 -req -passin pass:aivillage -days 365 -in server_keys/server.csr -CA server_keys/ca.crt -CAkey server_keys/ca.key -set_serial 01 -out server_keys/server.crt

echo Remove passphrase from server key:
openssl rsa -passin pass:aivillage -in server_keys/server.key -out server_keys/server.key

echo Generate client key
openssl genrsa -passout pass:aivillage -des3 -out client_keys/client.key 4096

echo Generate client signing request:
openssl req -passin pass:aivillage -new -key client_keys/client.key -out client_keys/client.csr -subj  "/C=US/ST=MD"

echo Self-sign client certificate:
openssl x509 -passin pass:aivillage -req -days 365 -in client_keys/client.csr -CA server_keys/ca.crt -CAkey server_keys/ca.key -set_serial 01 -out client_keys/client.crt

echo Remove passphrase from client key:
openssl rsa -passin pass:aivillage -in client_keys/client.key -out client_keys/client.key