import numpy
import zmq
import zlib, pickle

def initPusher(port):
    context = zmq.Context()
    sock = context.socket(zmq.PUSH)
    address = 'tcp://127.0.0.1:{}'.format(port)
    sock.bind(address)
    return sock

def initPuller(port):
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    address = 'tcp://127.0.0.1:{}'.format(port)
    sock.connect(address)
    return sock

def initPublisher(port):
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    address = 'tcp://127.0.0.1:{}'.format(port)
    sock.bind(address)
    return sock

def initSubsciber(port, topic):
    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    # sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
    sock.setsockopt(zmq.SUBSCRIBE, "".encode())
    address = 'tcp://127.0.0.1:{}'.format(port)
    sock.connect(address)
    return sock

def sendArray(socket, A, flags=0, copy=False, track=False):
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recvArray(socket, flags=0, copy=False, track=False):
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def pubSendArray(socket, topic, A, flags=0, copy=False, track=False):
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_string(topic, flags=zmq.SNDMORE)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def subRecvArray(socket, flags=0, copy=False, track=False):
    socket.recv_string()
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def sendZippedPickle(socket, topic, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)

def recvZippedPickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)