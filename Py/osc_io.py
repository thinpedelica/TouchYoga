from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.dispatcher import Dispatcher
from threading import Event, Semaphore, Thread

from logging import getLogger

logger = getLogger(__name__)

class OscIO:
    IP = '127.0.0.1'
    RECV_PORT = 8080
    SEND_PORT = 8090

    POINTS_LABEL = "/points"
    POSE_LABEL   = "/pose"
    STOP_LABEL   = "/stop"

    def __init__(self):
        self._is_running = True
        self._points     = []

        dispatcher = Dispatcher()
        dispatcher.map(self.POINTS_LABEL, self._points_handler)
        dispatcher.map(self.STOP_LABEL,   self._stop_handler)

        self._server = osc_server.ThreadingOSCUDPServer((self.IP, self.RECV_PORT), dispatcher)
        self._thread = Thread(target=self._recv_thread)
        self._thread.start()

        self._client = udp_client.UDPClient(self.IP, self.SEND_PORT)

        self._event = Event()
        self._semphore = Semaphore()

    def recv(self, timeout_sec=0.1):
        if not self._is_running:
            return (False, None)

        points = []

        ret = self._event.wait(timeout_sec)
        if ret:
            self._event.clear()
            with self._semphore:
                points = self._points

        return (True, points)

    def send(self, pose_id, pose_proba):
        if not self._is_running:
            return False

        msg = OscMessageBuilder(address=self.POSE_LABEL)
        msg.add_arg(pose_id)
        msg.add_arg(pose_proba)
        m = msg.build()

        self._client.send(m)
        return True

    def close(self):
        self._is_running = False
        self._server.shutdown()
        self._thread.join()

    def _recv_thread(self):
        self._server.serve_forever()

    def _points_handler(self, addr, *args):
        if not len(args) == 28:
            return

        with self._semphore:
            self._points = args
            self._event.set()

    def _stop_handler(self, addr):
        self._is_running = False
        self._server.shutdown()
