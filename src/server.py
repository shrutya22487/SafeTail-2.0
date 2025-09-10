import random
import numpy as np
import pickle
import pandas as pd

class Server:
    def __init__(self):
        """
        Initialize the Server by loading required models and data.
        """
        self.ping_data = pickle.load(open("data/ping_data.pkl", "rb"))
        self.yolo_regressor = pickle.load(open("models/yolo_thread_mlp_regressor_model.pkl", "rb"))
        self.yolo_scaler = pickle.load(open("models/yolo_thread_scaler.pkl", "rb"))
        self.yolo_resolutions = self.get_yolo_resolutions()
        self.noise_regressor = pickle.load(open("models/noise_mlp_regressor_model.pkl", "rb"))
        self.noise_scaler = pickle.load(open("models/noise_scaler.pkl", "rb"))
        self.instance_regressor = pickle.load(open("models/instance_mlp_regressor_model.pkl", "rb"))
        self.instance_scaler = pickle.load(open("models/instance_scaler.pkl", "rb"))

    def get_yolo_resolutions(self):
        """
        Load YOLO resolutions from a CSV file.
        Returns:
            list: List of resolution strings.
        """
        resolution_df  = pd.read_csv('./aman/ut_dqn/inst2.csv', header=None, names=range(12))
        
        return [resolution_df.iloc[i,0].split()[3] for i in range(len(resolution_df))]
    
    def get_propogation_delay(self, num):
        """
        Get a random propagation delay for a given node.
        Args:
            num (int): Node index (1-based).
        Returns:
            float: Randomly selected propagation delay.
        """
        return random.choice(self.ping_data[num-1])
    
    def get_tramission_delay(self, message_size , uplink , downlink):
        """
        Calculate the transmission delay for a message.
        Args:
            message_size (float): Size of the message in bytes.
            uplink (float): Uplink bandwidth in Mbps.
            downlink (float): Downlink bandwidth in Mbps.
        Returns:
            float: Transmission delay in seconds.
        """
        message_size = 8 * message_size
        uplink = uplink * 1000
        downlink = downlink * 1000 
        return (message_size / uplink) + (message_size / downlink)
    
    def get_computation_delay(self, state,node,task):
        """
        Estimate computation delay for a given node and task using ML models.
        Args:
            state (dict): Current state.
            node (int): Node index.
            task (str): Task type ('yolo', 'noise', 'instance').
        Returns:
            float or np.ndarray: Predicted computation delay.
        """
        
        def get_random_resolution(resolutions):
            """
            Select a random resolution from the list and compute its area.
            Args:
                resolutions (list): List of resolution strings (e.g., '640x480').
            Returns:
                int: Area of the selected resolution.
            """
            res = random.choice(resolutions)
            return int(res.split('x')[0]) * int(res.split('x')[1])
        
        if task == "yolo":
            number_of_users = state['LOAD'][node]
            if "RESOLUTION" not in state.keys():
                resolution = get_random_resolution(self.yolo_resolutions)
            else:
                resolution = state['RESOLUTION']

            inp = np.array([number_of_users, resolution])
            X_test = np.array([[inp[0],inp[1]]])
            X_test = np.array(X_test)
            st_dev = [2.7965017993200005, 6.393525858045778, 5.515155335473335, 7.100407896557774, 6.909932092510028, 7.124802705029803, 7.972889920066626, 9.872241111272556, 8.936860060306417, 12.450634470736823, 12.444299205142089, 11.830614817831743, 14.73346686870066, 17.763701566033472, 14.910366835514143, 19.22145836272576, 17.4644685302588, 19.87528975084137, 23.439168758290045, 20.457122880591005]
            pred = self.yolo_regressor.predict(self.yolo_scaler.transform(X_test))[0]
            return (pred + np.random.normal(0 , st_dev[int(inp[0])-1] ,1))/1000
        
        if task == "noise":
            inp = np.array([state['LOAD'][node], state['MESSAGE_SIZE']])
            X = np.array([inp])
            X = self.noise_scaler.transform(X)
            st_dev = [0.025221,0.027035,0.027473,0.029890,0.031513]
            predict = self.noise_regressor.predict(X)
            noise = np.random.normal(0, st_dev[state['LOAD'][node]-1], len(predict))
            predict_new = predict + abs(noise)/2
            return predict_new
        
        if task == "instance":
            number_of_users = state['LOAD'][node]
            size = state['MESSAGE_SIZE']
            X = np.array([[size, number_of_users]])
            X = self.instance_scaler.transform(X)
            std = [4.182686523232169, 27.4826173554251, 35.12740655487459, 39.941942069458854, 40.001337579530286]
            predict = self.instance_regressor.predict(X)
            noise = np.random.normal(0, std[number_of_users-1], len(predict))
            predict_new = predict + abs(noise)/2
            return predict_new
