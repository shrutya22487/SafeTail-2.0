import agent
import user
import servers
from controller import Controller

def main():
    users  = user.Users(number_of_requests=10)
    # TODO(@theshamiksinha) Should we pass the requests one by one or all at once to the controller?
    controller = Controller(requests=users.requests)
    controller.run()
    # TODO (@medhakashyap): Add any other code like graphs or anything here or in the controller 
