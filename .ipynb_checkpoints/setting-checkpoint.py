# setting.py

UTILIZATION_THRESHOLD = 0.9

F_peak = [ #main
    5*100,   #100 GFLOPS
    2*1000,  #1000 GFLOPS
    4*10000  #10000 GFLOPS
]

Bw_peak = [  #GB/s    #main
    25,      # device 0
    100,     # device 1
    200,     # device 2
]


P_peak = [ #main
    60,
    80,
    200
]


MIN_REQUIRED_ACCURACY = 0.3
MAX_REQUIRED_ACCURACY = 0.95

MIN_REQUIRED_ACCURACY_resNet = 0.69
MAX_REQUIRED_ACCURACY_resNet = 0.9

MIN_REQUIRED_ACCURACY_Bert = 0.74
MAX_REQUIRED_ACCURACY_Bert = 0.8

MIN_RESPONSE_TIME = 1   # seconds
MAX_RESPONSE_TIME = 50   # seconds


TIME_SLOT = 10          # decision period length (seconds)
NUM_REQUEST = 10          # maximum number of individual requests per set
MIN_NUM_REQUEST = 5
MAX_NUM_REQUEST = 20


