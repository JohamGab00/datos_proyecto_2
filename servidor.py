import socket
import threading

# Función para manejar las conexiones de los clientes
def handle_client(client_socket):
    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break

        with file_lock:
            with open(file_path, 'w') as file:
                file.write(data)
                print(f"Cliente agregó al archivo: {data}")

    client_socket.close()

# Función para que el servidor pueda escribir en el archivo
def server_write():
    while True:
        data = input("Escribe algo para agregar al archivo (o 'salir' para finalizar): ")

        if data == 'salir':
            break

        with file_lock:
            with open(file_path, 'w') as file:
                file.write(data)
                print(f"Servidor agregó al archivo: {data}")

# Configura el servidor
HOST = '192.168.100.231'  # Cambia por la dirección IP de tu servidor
PORT = 4000  # Cambia el puerto a un número disponible en tu máquina

# Crea un socket del servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)  # Permite hasta 5 conexiones en espera

print(f"Servidor escuchando en {HOST}:{PORT}")

# Ruta del archivo a editar
file_path = 'data.txt'
file_lock = threading.Lock()

# Inicia un hilo para que el servidor pueda escribir en el archivo
server_writer = threading.Thread(target=server_write)
server_writer.start()

while True:
    client_socket, client_address = server_socket.accept()
    print(f"Conexión establecida desde {client_address}")

    client_handler = threading.Thread(target=handle_client, args=(client_socket,))
    client_handler.start()
