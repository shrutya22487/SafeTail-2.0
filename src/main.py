import agent
import user
import servers
from controller import Controller

def main():
    users  = user.Users(number_of_requests=10)
    server_list = [servers.Server(server_index) for server_index in range(1, 6)]
    controller = Controller(requests=users.requests, server_list=server_list)
    controller.run()
    # TODO (@medhakashyap): Add any other code like graphs or anything here or in the controller 
