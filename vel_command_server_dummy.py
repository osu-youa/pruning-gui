import socket
import numpy as np
import time


def process():
    time.sleep(np.random.uniform(0.01, 0.02))
    return np.random.randint(0, 20, 5).astype(np.uint8)


if __name__ == '__main__':

    ADDRESS = 'localhost'
    PORT = 10001

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = (ADDRESS, PORT)
    try:
        print('Starting up server on {}:{}'.format(*address))
        sock.bind(address)
        sock.listen(1)
        while True:
            print('Waiting for connection')
            connection, client_address = sock.accept()
            print('Connection accepted!')
            try:
                # Keep accepting commands until the connection is closed
                while True:
                    # Wait for client to send a message, even if it's purely for synchronization
                    master_buffer = connection.recv(1024)
                    received_array = np.frombuffer(master_buffer, dtype=np.float64)
                    print(received_array)
                    if not len(received_array):
                        print('Connection looks like it has been terminated...')
                        break

                    # Create an array and send it to the client
                    array = process()
                    connection.sendall(array.tobytes())
            except ConnectionAbortedError:
                print('Connection was aborted by the client!')
            finally:
                connection.close()
                print('Connection terminated, waiting for new connection...')
    finally:
        sock.close()