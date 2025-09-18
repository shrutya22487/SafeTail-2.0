 
class Request:
    def __init__(self, request_id):
        self.request_id = request_id
        self.process_id = None
        self.ram_usage = None
        self.cpu_usage = None

# TODO(@BingoBoy479): Add state here or a corresponding class edit in all other files accordingly
class Users:
    def __init__(self, number_of_requests):
        self.number_of_requests = number_of_requests
        self.requests = [Request(i) for i in range(number_of_requests)]
    

   