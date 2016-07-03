from socket import gethostname

def is_host():
    return gethostname() != "whz4"
