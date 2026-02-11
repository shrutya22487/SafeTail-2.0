from user import Request
from servers import Server
import numpy as np
req = Request(request_id=0, process_id=0, combination='s', message_size=1024, bandwidth=20, load=np.array((0,0,0,0,0)))
server = Server(server_index=5)
# server.schedule_request(req)
req.populate_request_from_csv(5, 'dds')