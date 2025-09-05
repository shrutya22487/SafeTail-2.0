class Server:
    def __init__(self, 
                 cpu_clock_speed, 
                 total_ram, 
                 gpu_clock_speed, 
                 gpu_memory, 
                 num_cores,
                 uplink_bw, 
                 downlink_bw, 
                 mem_util, 
                 cpu_util, 
                 active_users, 
                 avail_ram,
                 current_load, 
                 cores_used, 
                 core_freq, 
                 bg_cpu_util, 
                 gpu_freq, 
                 gpu_util,
                 gpu_mem_avail, 
                 gpu_mem_util, 
                 max_load):
        """
        Initialize an Edge Server with both static and dynamic states.
        """

        # Static configuration
        self.cpu_clock_speed = cpu_clock_speed
        self.total_ram = total_ram
        self.gpu_clock_speed = gpu_clock_speed
        self.gpu_memory = gpu_memory
        self.num_cores = num_cores

        # Dynamic configuration
        self.uplink_bw = uplink_bw
        self.downlink_bw = downlink_bw
        self.mem_util = mem_util
        self.cpu_util = cpu_util
        self.active_users = active_users
        self.avail_ram = avail_ram
        self.current_load = current_load
        self.cores_used = cores_used
        self.core_freq = core_freq
        self.bg_cpu_util = bg_cpu_util
        self.gpu_freq = gpu_freq
        self.gpu_util = gpu_util
        self.gpu_mem_avail = gpu_mem_avail
        self.gpu_mem_util = gpu_mem_util
        self.max_load = max_load

    def __repr__(self):
        return (f"EdgeServer(CPU={self.cpu_clock_speed}GHz, RAM={self.total_ram}GB, "
                f"GPU={self.gpu_clock_speed}GHz/{self.gpu_memory}GB, "
                f"Cores={self.num_cores}, Load={self.current_load}/{self.max_load})")
