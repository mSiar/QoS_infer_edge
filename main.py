
import random
import this
import time
import numpy as np
import pandas as pd
import math
import copy
import csv
from itertools import islice
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from collections import namedtuple
import heapq
import itertools
from itertools import product
from scipy.optimize import nnls
from setting import (
    NUM_REQUEST,
    MIN_NUM_REQUEST,
    MAX_NUM_REQUEST,
    UTILIZATION_THRESHOLD,
    MIN_REQUIRED_ACCURACY,
    MIN_REQUIRED_ACCURACY_resNet,
    MAX_REQUIRED_ACCURACY,
    MAX_REQUIRED_ACCURACY_resNet,
    MAX_REQUIRED_ACCURACY_Bert,
    MIN_REQUIRED_ACCURACY_Bert,
    MIN_RESPONSE_TIME,
    MAX_RESPONSE_TIME,
    TIME_SLOT,
    F_peak,
    P_peak,
    Bw_peak)



class MLModel:

    def __init__(self, id: int, name: str, required_flops: float, required_bw: float, accuracy: float):
        self.id = id
        self.name = name
        self.required_flops = required_flops
        self.required_bw = required_bw
        self.accuracy = accuracy



class EdgeNode:

    def __init__( self, id, flops_capacity, energy_limit, FlopsWatt, f_peak, bw_peak, p_idle, models):
        self.id = id
        self.total_flops_capacity = flops_capacity
        self.total_energy_limit = energy_limit
        self.available_flops = self.total_flops_capacity * UTILIZATION_THRESHOLD
        self.available_energy = self.total_energy_limit
        self.replicas = []
        self.models = models 
        self.total_util_track = []
        self.F_peak = f_peak
        self.Bw_peak = bw_peak
        self.p_idle = p_idle
        self.FlopsWatt = FlopsWatt
        self.busy_until = 0.0
        self.dynamic_energy_used = 0.0
        self.static_energy_used = 0.0

        
    def service_time(self, model: "MLModel"):
        eta_comp = 1
        eta_bw = 1
        T_comp = round(model.required_flops/(eta_comp*self.available_flops), 5)
        T_mem = round(model.required_bw/(eta_bw*self.Bw_peak), 5)
        srv_time = max(T_comp , T_mem)
        srv_time = round(srv_time, 5)
        return srv_time
        


    def process_request(self,current_time: float, model:"MLModel"):
        exec_time = self.service_time(model)
        start_time = max(current_time, self.busy_until)
        finish_time = start_time + exec_time
        self.busy_until = finish_time
        E_dyn = self.energy_per_request(model)   
        self.consume_energy(E_dyn)
        return start_time, finish_time, model.accuracy, E_dyn


    def energy_per_request(self, model: "MLModel"):  
        comp_energy = model.required_flops/self.FlopsWatt
        alfa_mem = 0.5 
        mem_energy = alfa_mem*model.required_bw
        E_dyn = comp_energy+mem_energy   
        E_dyn = round(E_dyn, 3)
        return  E_dyn


    def consume_energy(self, e_dyn: float):
        self.dynamic_energy_used += e_dyn
        self.available_energy = max(0.0, self.available_energy - e_dyn) 

    
    def current_energy_usage(self):
        return self.dynamic_energy_used + self.static_energy_used 

    
    def used_energy(self):
        return self.total_energy_limit - self.available_energy

    
    def under_energy_limit(self):
        return self.available_energy > 0.0


class Request_set:
    
    def __init__(self,
                 id,
                 arrival_time,       
                 qos_accuracy,
                 qos_response_time,
                 num_completed_request,
                 estimated_accuracy,
                 finish_time,
                 flag):
        
        self.id = id
        self.arrival_time = arrival_time
        self.num_completed_request = num_completed_request
        self.estimated_accuracy = estimated_accuracy
        self.finish_time = finish_time
        self.required_accuracy = qos_accuracy
        self.qos_response_time = qos_response_time
        self.flag = flag

        self.arrival_times: list[float] = []
        self.requests: list[Request] = []
        self.next_req_index: int = 0

    def init_requests(self, arrival_times: list[float], num_requests: int):
        if len(arrival_times) < num_requests:
            arrival_times = list(arrival_times) + [arrival_times[-1]] * (num_requests - len(arrival_times))
        elif len(arrival_times) > num_requests:
            arrival_times = list(arrival_times[:num_requests])

        self.arrival_times = arrival_times
        self.requests = []

        for k in range(num_requests):
            t_arr = self.arrival_times[k]
            deadline = self.qos_response_time     
            req = Request(
                set_id=self.id,
                req_id=k + 1,
                arrival_time=t_arr,
                deadline=deadline,
                min_accuracy=self.required_accuracy,
            )
            self.requests.append(req)
        self.next_req_index = 0

    def get_next_request(self):
        if self.next_req_index >= len(self.requests):
            return None
        return self.requests[self.next_req_index]

    def mark_request_completed(self):
        if self.next_req_index < len(self.requests):
            self.next_req_index += 1

    def count_satisfied(self):
        return sum(1 for r in self.requests if r.is_qos_satisfied())

    def total_dynamic_energy(self):
        return sum(r.energy_dyn for r in self.requests)


class Request: 
    def __init__(self,
                 set_id: int,
                 req_id: int,
                 arrival_time: float,
                 deadline: float,
                 min_accuracy: float):
        self.set_id = set_id         
        self.req_id = req_id          
        self.arrival_time = arrival_time
        self.deadline = arrival_time+deadline
        self.min_accuracy = min_accuracy
        self.edge_node_id = None
        self.model_index = None
        self.model_name = None
        self.start_time = None
        self.finish_time = None
        self.energy_dyn = 0.0
        self.met_deadline = None
        self.met_accuracy = None

    
    def record_execution(self,   
                         edge_node_id: int,
                         model_name: str,
                         start_time: float,
                         finish_time: float,
                         energy_dyn: float,
                         achieved_accuracy: float):

        self.edge_node_id = edge_node_id
        self.model_name = model_name
        self.start_time = start_time
        self.finish_time = finish_time
        self.energy_dyn = energy_dyn
        self.met_deadline = ( self.finish_time <= self.deadline)
        self.met_accuracy = (achieved_accuracy >= self.min_accuracy)

        if self.met_deadline and self.met_accuracy:
            self.status = "succeeded"
            self.fail_reason = None
        else:
            self.status = "failed"
            self.fail_reason = "qos"

    def mark_failed(self, reason: str):
        self.status = "failed"
        self.fail_reason = reason
        self.met_deadline = False
        self.met_accuracy = False

    def is_qos_satisfied(self):
        return bool(self.met_deadline and self.met_accuracy)
        

                                                    
class DecisionMaker_energy_priority:
    def __init__(self):
        pass

    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs, response_logs2, nodes_util, nodes_energy, current_time=None):
        init_busy = [node.busy_until for node in edge_nodes]
        num_input_req = sum([len(rq.requests) for rq in request_sets])
        cur_eng = []
        true_in = sum(len(rs.requests) - rs.num_completed_request for rs in request_sets)

        if current_time is None:
            current_time = 0.0
        num_completed = 0
        rhp_max = UTILIZATION_THRESHOLD
        current_util = [0 for _ in range(len(edge_nodes))]
        curr_all_eng = [0 for _ in range(len(edge_nodes))]
        curr_dyn_eng = [0 for _ in range(len(edge_nodes))]
        curr_stc_eng = [0 for _ in range(len(edge_nodes))]
        moel_container = []
        cc = 0
        for node in edge_nodes:
            for model in models:
                E_dyn = node.energy_per_request(model)
                service_time = node.service_time(model)
                energy_metric = E_dyn
                moel_container.append({
                    "edge_id": node.id,            
                    "model": model,
                    "accuracy": model.accuracy,
                    "service_time": service_time,
                    "energy": energy_metric,       
                    "dynamic_energy": E_dyn,        
                })

        moel_container.sort(key=lambda r:r['dynamic_energy']) 
        node_completed = [0 for nd in range(len(edge_nodes))]
        for ctn in (moel_container):
            node_id = ctn["edge_id"]
            node = next(n for n in edge_nodes if n.id == node_id)
            matching_requests = []

            for r in request_sets:
                if (r.next_req_index < len(r.requests) and round(sum(r.estimated_accuracy) + ctn["accuracy"], max(decimal_places(r.required_accuracy) , decimal_places(ctn["accuracy"])))>= round( r.required_accuracy * (r.num_completed_request + 1), decimal_places(r.required_accuracy))):
                    matching_requests.append(r)

            if not len(matching_requests)>0: 
                continue
                    
            arriv_time = []
            inter_arv = []
            arriv_rate = []
            flag = False
            arv_id = 0
            
            for r_set in matching_requests:
                
                total_reqs = len(r_set.requests)
                remaining = total_reqs - r_set.num_completed_request
                if remaining <= 0:
                    continue
                    
                for _ in range(remaining):
                    req_obj = r_set.get_next_request()
                    if req_obj is None:
                        break
                         
                    util_increment = ctn["service_time"] / TIME_SLOT  
                    if current_util[node.id] + util_increment > UTILIZATION_THRESHOLD:   
                        continue

                    model = ctn["model"]
                    energy_needed = ctn["dynamic_energy"]
                    if node.available_energy < energy_needed:  
                        continue


                    exec_time = node.service_time(model)       
                    real_start = max(node.busy_until, req_obj.arrival_time)
                    
                    real_finish = real_start + exec_time
                    qos_deadline_ok = (
                        round(real_finish, decimal_places(r_set.qos_response_time))
                        <=
                        round(req_obj.arrival_time + r_set.qos_response_time, decimal_places(r_set.qos_response_time) ))
            
                    if len(r_set.estimated_accuracy) > 0:
                        sum_est_acc = sum(r_set.estimated_accuracy)
                    else:
                        sum_est_acc = 0.0
            
                    qos_accuracy_ok = (round(ctn["accuracy"] + sum_est_acc,
                              max(decimal_places(ctn["accuracy"]), decimal_places(r_set.required_accuracy))) >=
                        round(r_set.required_accuracy * (r_set.num_completed_request + 1),
                              decimal_places(r_set.required_accuracy)))

                    if not (qos_deadline_ok and qos_accuracy_ok): 
                        continue

                    start, finish, acc_real, E_dyn_real = node.process_request(req_obj.arrival_time, model)

                    req_obj.record_execution(
                        edge_node_id = node.id,
                        model_name= model.name,
                        start_time=start,
                        finish_time=finish,
                        energy_dyn=E_dyn_real,
                        achieved_accuracy=acc_real,
                    )

                    response_time = finish - req_obj.arrival_time
                    completed_requests.append((req_obj.arrival_time, start, finish, node.id, model.name))
                    
                    response_logs.append({
                        "request_set_id": r_set.id,
                        "arrival_time": req_obj.arrival_time,
                        "start_time": start,
                        "finish_time": finish,
                        "response_time": response_time,
                        "accuracy_improvement": acc_real - r_set.required_accuracy,
                        "model": model.name,
                        "edge_node_id": node.id,
                        "reason": "success",
                    })
                    r_set.estimated_accuracy.append(acc_real)
                    r_set.finish_time = max(r_set.finish_time, finish)
                    r_set.num_completed_request +=1
                    num_completed += 1
                    node_completed[node.id]+=1
                    curr_all_eng[node.id]+= E_dyn_real
                    curr_dyn_eng[node.id]+=E_dyn_real

                    r_set.mark_request_completed()            
                    current_util[node.id] += util_increment

        remain = sum(len(rs.requests) - rs.num_completed_request for rs in request_sets)

        for node in edge_nodes:
            n_nodes = len(F_peak)
            idle_energy = node.p_idle*TIME_SLOT   
            node.static_energy_used += idle_energy
            node.available_energy = max(0.0, node.available_energy - idle_energy)
            curr_all_eng[node.id]+= idle_energy
            curr_stc_eng[node.id]+=idle_energy

        nodes_util.append(list(current_util))
        nodes_energy.append([nd.current_energy_usage() for nd in (edge_nodes)])
        arrivals_node = defaultdict(lambda: defaultdict(list))

        for (arrival, start, finish, node_id, model_name) in completed_requests:
            arrivals_node[node_id][model_name].append(arrival)

        response_logs2.append({
            "num_input_request": true_in,
            "num_completed_request": num_completed,
            "nodes_utilization": current_util,
            "total_energy_usage":  list(nodes_energy[-1]) ,
            "currnt_all_energy_used": curr_all_eng,
            "current_dyn_energy_used": curr_dyn_eng,
            "current_stc_energy_used": curr_stc_eng,
        }  
        )




class DecisionMaker_edgeParams_request_priority_MlFirst:
    def __init__(self):
        pass
        
    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs, response_logs2, nodes_util, nodes_energy, current_time=None):
        if current_time is None:
            current_time = 0.0

        rhp_max = UTILIZATION_THRESHOLD
        init_busy = [node.busy_until for node in edge_nodes]
        true_in = sum(len(rs.requests) - rs.num_completed_request for rs in request_sets)

        current_util = [0 for _ in range(len(edge_nodes))]
        curr_all_eng = [0 for _ in range(len(edge_nodes))]
        curr_dyn_eng = [0 for _ in range(len(edge_nodes))]
        curr_stc_eng = [0 for _ in range(len(edge_nodes))]
        num_completed=0
        for i in range(len(edge_nodes)):
            current_util[i] = 0
        
        moel_container = []
        cc = 0
        for node in edge_nodes:
            for model in models:
                E_dyn = node.energy_per_request(model)
        
                if node.available_energy < E_dyn:
                    continue

                service_time = node.service_time(model)
        
                energy_metric = E_dyn
        
                moel_container.append({
                    "edge_id": node.id,             
                    "model": model,
                    "accuracy": model.accuracy,
                    "service_time": service_time,
                    "energy": energy_metric,        
                    "dynamic_energy": E_dyn,        
                })

        moel_container.sort(key=lambda r: (r["service_time"], r["dynamic_energy"], -r["accuracy"]))
        request_sets.sort(
            key=lambda r: r.qos_response_time
        )
        
        for ctn in (moel_container):
            node_id = ctn["edge_id"]
            node = next(n for n in edge_nodes if n.id == node_id)
            matching_requests = []

            for r in request_sets:
                if (r.next_req_index < len(r.requests) and round(sum(r.estimated_accuracy) + ctn["accuracy"], max(decimal_places(r.required_accuracy) , decimal_places(ctn["accuracy"])))>= round( r.required_accuracy * (r.num_completed_request + 1), decimal_places(r.required_accuracy))):
                    matching_requests.append(r)

            if not len(matching_requests)>0:
                continue

            arriv_time = []
            inter_arv = []
            arriv_rate = []
            flag = False
            arv_id = 0
            
            for r_set in request_sets: 
                
                total_reqs = len(r_set.requests)
                remaining = total_reqs - r_set.num_completed_request
                if remaining <= 0:
                    continue
                    
                for _ in range(remaining):
                    req_obj = r_set.get_next_request()
                    if req_obj is None:
                        break
                        
                    util_increment = ctn["service_time"] / TIME_SLOT   
                    if current_util[node.id] + util_increment > UTILIZATION_THRESHOLD:
                        continue

                    model = ctn["model"]
                    energy_needed = ctn["dynamic_energy"]
                    if node.available_energy < energy_needed:
                        continue
                    
                    
                    exec_time = node.service_time(model)      
                    real_start = max(node.busy_until, req_obj.arrival_time)
                    
                    real_finish = real_start + exec_time

                    qos_deadline_ok = (
                        round(real_finish, decimal_places(r_set.qos_response_time))
                        <=
                        round(req_obj.arrival_time + r_set.qos_response_time, decimal_places(r_set.qos_response_time) ))
            
                    if len(r_set.estimated_accuracy) > 0:
                        sum_est_acc = sum(r_set.estimated_accuracy)
                    else:
                        sum_est_acc = 0.0
            
                    qos_accuracy_ok = (round(ctn["accuracy"] + sum_est_acc,
                              max(decimal_places(ctn["accuracy"]), decimal_places(r_set.required_accuracy))) >=
                        round(r_set.required_accuracy * (r_set.num_completed_request + 1),
                              decimal_places(r_set.required_accuracy)))

                    if not (qos_deadline_ok and qos_accuracy_ok):
                        req_obj.mark_failed("qos")
                        continue

                    start, finish, acc_real, E_dyn_real = node.process_request(req_obj.arrival_time, model)

                    req_obj.record_execution(
                        edge_node_id = node.id,
                        model_name= model.name,
                        start_time=start,
                        finish_time=finish,
                        energy_dyn=E_dyn_real,
                        achieved_accuracy=acc_real,
                    )

                    response_time = finish - req_obj.arrival_time
                    completed_requests.append((req_obj.arrival_time, start, finish, node.id, model.name))
                    
                    response_logs.append({
                        "request_set_id": r_set.id,
                        "arrival_time": req_obj.arrival_time,
                        "start_time": start,
                        "finish_time": finish,
                        "response_time": response_time,
                        "accuracy_improvement": acc_real - r_set.required_accuracy,
                        "model": model.name,
                        "edge_node_id": node.id,
                        "reason": "success",
                    })
                    r_set.estimated_accuracy.append(acc_real)
                    r_set.finish_time = max(r_set.finish_time, finish)
                    r_set.num_completed_request +=1
                    r_set.mark_request_completed()
                    num_completed+=1
            
                    current_util[node.id] += util_increment
                    curr_all_eng[node.id]+= E_dyn_real
                    curr_dyn_eng[node.id]+=E_dyn_real
        
        for node in edge_nodes:
            idle_energy = node.p_idle* TIME_SLOT  
            node.static_energy_used += idle_energy
            node.available_energy = max(0.0, node.available_energy - idle_energy)

            curr_all_eng[node.id]+= idle_energy
            curr_stc_eng[node.id]+=idle_energy

        nodes_util.append(list(current_util))
        nodes_energy.append([nd.current_energy_usage() for nd in (edge_nodes)])

        response_logs2.append({
            "num_input_request": true_in,
            "num_completed_request": num_completed,
            "nodes_utilization": current_util,
            "total_energy_usage":  list(nodes_energy[-1]) ,
            "currnt_all_energy_used": curr_all_eng,
            "current_dyn_energy_used": curr_dyn_eng,
            "current_stc_energy_used": curr_stc_eng,
        } ) 




class DecisionMaker_edgeParams_request_priority_reqFirst:
    def __init__(self):
        pass
        
    
    def decide_allocation(self, request_sets, edge_nodes, models, completed_requests, response_logs, response_logs2, nodes_util, nodes_energy, current_time=None):

        init_busy = [node.busy_until for node in edge_nodes]
        if current_time is None:
            current_time = 0.0

        true_in = sum(len(rs.requests) - rs.num_completed_request for rs in request_sets)
        num_completed = 0
        rhp_max = UTILIZATION_THRESHOLD
        current_util = [0 for _ in range(len(edge_nodes))]
        curr_all_eng = [0 for _ in range(len(edge_nodes))]
        curr_dyn_eng = [0 for _ in range(len(edge_nodes))]
        curr_stc_eng = [0 for _ in range(len(edge_nodes))]
        moel_container = []
        cc = 0
        for node in edge_nodes:
            for model in models:
                E_dyn = node.energy_per_request(model)
                service_time = node.service_time(model)
                energy_metric = E_dyn
                moel_container.append({
                    "edge_id": node.id,          
                    "model": model,
                    "accuracy": model.accuracy,
                    "service_time": service_time,
                    "energy": energy_metric,      
                    "dynamic_energy": E_dyn,       
                })
            
        moel_container.sort(key=lambda r: (r["service_time"], r["dynamic_energy"], -r["accuracy"]))
        request_sets.sort(
                key=lambda r: r.qos_response_time
            )
        node_completed = [0 for nd in range(len(edge_nodes))]
        for req_set in request_sets: 
            total_reqs = len(req_set.requests)
            remaining = total_reqs - req_set.num_completed_request
            if remaining <= 0:
                continue
                
            for _ in range(remaining):
                req_obj = req_set.get_next_request()
                if req_obj is None:
                    break 
                
                for ctn in (moel_container):
                    node_id = ctn["edge_id"]
                    node = next(n for n in edge_nodes if n.id == node_id)     
                    arriv_time = []
                    inter_arv = []
                    arriv_rate = []
                    flag = False
                    arv_id = 0
                    util_increment = ctn["service_time"] / TIME_SLOT  
                    
                    if current_util[node.id] + util_increment > UTILIZATION_THRESHOLD:   
                        continue
    
                    model = ctn["model"]
    
                    energy_needed = ctn["dynamic_energy"]
                    if node.available_energy < energy_needed: 
                        continue
    
                    exec_time = node.service_time(model)    
                    real_start = max(node.busy_until, req_obj.arrival_time)
                    real_finish = real_start + exec_time
                    qos_deadline_ok = (
                        round(real_finish, decimal_places(req_set.qos_response_time))
                        <=
                        round(req_obj.arrival_time + req_set.qos_response_time, decimal_places(req_set.qos_response_time) ))
            
                    if len(req_set.estimated_accuracy) > 0:
                        sum_est_acc = sum(req_set.estimated_accuracy)
                    else:
                        sum_est_acc = 0.0
            
                    qos_accuracy_ok = (round(ctn["accuracy"] + sum_est_acc,
                              max(decimal_places(ctn["accuracy"]), decimal_places(req_set.required_accuracy))) >=
                        round(req_set.required_accuracy * (req_set.num_completed_request + 1),
                              decimal_places(req_set.required_accuracy)))
    
                    if not (qos_deadline_ok and qos_accuracy_ok):  
                        continue
    
                    start, finish, acc_real, E_dyn_real = node.process_request(req_obj.arrival_time, model)
                    req_obj.record_execution(
                        edge_node_id = node.id,
                        model_name= model.name,
                        start_time=start,
                        finish_time=finish,
                        energy_dyn=E_dyn_real,
                        achieved_accuracy=acc_real,
                    )
                    
                    response_time = finish - req_obj.arrival_time
                    completed_requests.append((req_obj.arrival_time, start, finish, node.id, model.name))
                    response_logs.append({
                        "request_set_id": req_set.id,
                        "arrival_time": req_obj.arrival_time,
                        "start_time": start,
                        "finish_time": finish,
                        "response_time": response_time,
                        "accuracy_improvement": acc_real - req_set.required_accuracy,
                        "model": model.name,
                        "edge_node_id": node.id,
                        "reason": "success",
                    })
                    req_set.estimated_accuracy.append(acc_real)
                    req_set.finish_time = max(req_set.finish_time, finish)
                    req_set.num_completed_request +=1
                    num_completed += 1
                    node_completed[node.id]+=1
                    req_set.mark_request_completed()            
                    current_util[node.id] += util_increment
                    curr_all_eng[node.id]+= E_dyn_real
                    curr_dyn_eng[node.id]+=E_dyn_real
                    break

            
        remain = sum(len(rs.requests) - rs.num_completed_request for rs in request_sets)
        for node in edge_nodes:
            n_nodes = len(F_peak)
            idle_energy = node.p_idle*TIME_SLOT 
            node.static_energy_used += idle_energy
            node.available_energy = max(0.0, node.available_energy - idle_energy)
            curr_all_eng[node.id]+= idle_energy
            curr_stc_eng[node.id]+=idle_energy


        nodes_util.append(list(current_util))
        nodes_energy.append([nd.current_energy_usage() for nd in (edge_nodes)])
        arrivals_node = defaultdict(lambda: defaultdict(list))
        for (arrival, start, finish, node_id, model_name) in completed_requests:
            arrivals_node[node_id][model_name].append(arrival)

        response_logs2.append({
            "num_input_request": true_in,
            "num_completed_request": num_completed,
            "nodes_utilization": current_util,
            "total_energy_usage":  list(nodes_energy[-1]) ,
            "currnt_all_energy_used": curr_all_eng,
            "current_dyn_energy_used": curr_dyn_eng,
            "current_stc_energy_used": curr_stc_eng,
        } ) 






class Simulator:
    def __init__(self, edge_nodes, models, duration, decision_maker):
        self.edge_nodes = edge_nodes
        self.models = models
        self.duration = duration
        self.request_sets = []
        self.completed_requests = []
        self.decision_maker = decision_maker
        self.decision_times = []
        self.response_logs = []
        self.response_logs2 = []
        self.nodes_util = []
        self.nodes_energy = []

    def run(self):
        decision_times = []
        start_sim_time = 0
        maximum_arrivals = max(st.arrival_time for st in self.request_sets)
        while(start_sim_time<=maximum_arrivals):
            current_requests = [] 
            current_requests = [st for st in self.request_sets if st.arrival_time>=start_sim_time and st.arrival_time<start_sim_time+TIME_SLOT]
            arrivals = [st.arrival_time for st in current_requests]
            num_arv = sum(len(rs.requests) for rs in current_requests)
            true_in = sum(len(rs.requests) - rs.num_completed_request for rs in current_requests)
            if current_requests: 
                start_decision = time.time()
                self.decision_maker.decide_allocation(current_requests, self.edge_nodes, self.models, self.completed_requests, self.response_logs, self.response_logs2, self.nodes_util, self.nodes_energy, start_sim_time)
                end_decision = time.time()
                decision_times.append((end_decision - start_decision))
            start_sim_time += TIME_SLOT
        self.decision_times = decision_times   
        
    def print_stats(self, approach, edge_num, arrival_rate, energy_interval, log):
        print(f"---------------------------  Approach:  {approach}  ---------------------------------")
        total = sum([len(st.requests) for st in self.request_sets])
        completed = sum([req.num_completed_request for req in self.request_sets])
        rejected = sum([len(req.requests)-req.num_completed_request for req in self.request_sets ])#if req.flag==True])
        delays =  [t[2] - t[0] for t in self.completed_requests]  
        diff_time = [(req.qos_response_time - req.finish_time) for req in self.request_sets]
        diff_avg_acc = [(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0] 
        total_diff_time = sum([((req.arrival_time+req.qos_response_time) - req.finish_time) for req in self.request_sets  if req.num_completed_request>0])
        total_diff_avg_acc = sum([(sum(req.estimated_accuracy)/req.num_completed_request)- req.required_accuracy for req in self.request_sets if req.num_completed_request>0])
        print(f"Total Requests: {total}")
        print(f"Completed Requests: {(completed)}")
        print(f"Rejected Requests: {(rejected)}")
        print(f"Difference between finish time of completed request and deadline, for all request sets:  ", total_diff_time)
        print(f"Difference between average accuracy of completed request and expected minimum accuracy, for request sets with completed requests:  ", total_diff_avg_acc)
        
        if (completed) > 0:
            avg_delay = sum(delays) / total
            percentile_90 = np.percentile(delays, 90)
            percentile_95 = np.percentile(delays, 95)
            print(f"Average Total Time per Request: {avg_delay:.2f} seconds")
            print(f"90th Percentile Response Time: {percentile_90:.2f} seconds")
            print(f"95th Percentile Response Time: {percentile_95:.2f} seconds")
        total_energy_used = sum(node.used_energy() for node in self.edge_nodes)
        print(f"Total Energy Consumed: {total_energy_used:.2f}")
        avg_total_util = 0.0
        T = len(self.nodes_util)
        N = len(self.edge_nodes)
        for j in range(N):
            s = 0.0
            for t in range(T):
                u = self.nodes_util[t]
                s += u[j]
            node_avg = s / T  
            print(f"Node {self.edge_nodes[j].id} Avg Utilization: {node_avg}")
            avg_total_util += node_avg
        
        avg_total_util = avg_total_util / N
        print(f"Average Utilization (all nodes): {avg_total_util}")
        
        if getattr(self, "decision_times", None):
            avg_decision_time = sum(self.decision_times) / len(self.decision_times)
            perc_90_decision = np.percentile(self.decision_times, 90)
            print(f"Average Decision Time per Slot: {avg_decision_time:.4f} seconds")
            print(f"90th Percentile Decision Time: {perc_90_decision:.4f} seconds")
        else: 
            print("Average Decision Time per Slot: N/A (no decisions made)")

        if log is not None:

            if energy_interval is None:
                energy_interval= [10**4, 10**6] # Defaulf interval
            log.append({
                        "approach": approach,
                        "edge_num": edge_num, 
                        "Arrival_rate": arrival_rate,
                        "Energy_interval": energy_interval,
                        "total_request": total,
                        "Completed_requests": completed,
                        "Rejected_requests": rejected,
                        "Speed:Deadline-Finish": total_diff_time,
                        "Acc:AvgAcc-ExpAcc": total_diff_avg_acc,
                        "Total_energy_used": total_energy_used,
                        "Total_utilisation": avg_total_util, 
                        "avg_runTime": avg_decision_time
                    })    

    def save_response_logs(self, approach):
        filename=f"{approach}_response_logs_ReqCompleted.csv"
        if not self.response_logs:
            print("No response logs to save.")
            return

        print("Number of response logs( for completed requests):", len(self.response_logs))
        with open(filename, mode='w', newline='') as csvfile:
            fieldnames = ["request_set_id", "arrival_time", "start_time", "finish_time", "response_time", "accuracy_improvement", "model",
                          "edge_node_id", "reason"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.response_logs:
                writer.writerow(entry)
        print(f"Saved response logs to {filename}")

    def save_response_logs_2(self, approach):
        filename=f"{approach}_response_logs_timeSlots.csv"
        if not self.response_logs2:
            print("No response logs to save.")
            return
        with open(filename, mode='w', newline='') as csvfile:
            fieldnames = ["num_input_request", "num_completed_request", "nodes_utilization", "total_energy_usage", "currnt_all_energy_used", "current_dyn_energy_used", "current_stc_energy_used"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.response_logs2:
                writer.writerow(entry)
        print(f"Saved response logs2 to {filename}")


def decimal_places(val):
    s = str(val)
    if '.' in s:
        return len(s.split('.')[-1])
    return 0

    
def main():
    
    DEFAULT_EDGE_NUM = 3
    DEFAULT_ARRIVAL = 1
    NUM_REQUEST = 10
    random.seed(42)
    np.random.seed(42)
    energy_scale = [random.randint(10,100) for i in range(DEFAULT_EDGE_NUM)] 
    ALPHA = [random.uniform(0.1, 0.5) for _ in range(DEFAULT_EDGE_NUM)]
    round_alpha = [round(a,3) for a in ALPHA]

    models = [
        MLModel(id=0, name="yolov8n", required_flops=8.7, required_bw = 12.8/1000.0,accuracy= 0.37),   # YOLO models
        MLModel(id=1, name="yolov8s", required_flops=28.6, required_bw = 44.8/1000.0, accuracy= 0.45),
        MLModel(id=2, name="yolov8m", required_flops=78.9, required_bw = 103.6/1000.0, accuracy= 0.5), 
        MLModel(id=3, name="yolov8l", required_flops=165.2, required_bw = 174.8/1000.0, accuracy= 0.53),
        MLModel(id=4, name="yolov8x", required_flops=257.8, required_bw = 272.8/1000.0,  accuracy= 0.54),
    ]

    # models = [
    # MLModel(id=0, name="ResNet18",  required_flops=1.8,  required_bw=46.8/1000.0, accuracy=0.69), #parameters=11.7M   #ResNet models
    # MLModel(id=1, name="ResNet34",  required_flops=3.6,  required_bw=87.2/1000.0, accuracy=0.73),  #parameters=21.8M
    # MLModel(id=2, name="ResNet50",  required_flops=4.1,  required_bw=102.4/1000.0, accuracy=0.761),  #parameters=25.6M
    # MLModel(id=3, name="ResNet101", required_flops=7.8,  required_bw=178/1000.0, accuracy=0.8),  ##parameters=44.5M
    # MLModel(id=4, name="ResNet152", required_flops=11.6, required_bw=240.8/1000.0, accuracy=0.9),  #parameters=60.2M#
    # ]

    # models = [
    # MLModel(id=0, name="TinyBERT-4L", required_flops=1.2,  required_bw=58/1000.0, accuracy=0.77),  # Params=14.5M   #BERT models
    # MLModel(id=1, name="TinyBERT-6L", required_flops=11.3, required_bw=268/1000.0, accuracy=0.8),  # Params=67.5M 
    # MLModel(id=2, name="DistilBERT-4L", required_flops=7.6,  required_bw=208/1000.0, accuracy=0.74),  # Params=52.2M 
    # MLModel(id=3, name="DistilBERT-6L", required_flops=11.3, required_bw=248.8/1000.0, accuracy=0.77),  # Params=66.6M 
    # MLModel(id=4, name="MobileBERT",      required_flops=5.7, required_bw=101.2/1000.0, accuracy=0.76),  # Params=25.3M 
    # MLModel(id=5, name="MobileBERTTINY",  required_flops=3.1, required_bw=60.4/1000.0, accuracy=0.75),  # Params=15.1M 
    # MLModel(id=6, name="BERT-Base",  required_flops=22.5, required_bw=436/1000.0, accuracy=0.8),  # Params=15.1M
    # ]



    def build_nodes(edge_num, scale_param):
        edge_nodes = []
        
        for i in range(edge_num):
            base_flops_capacity = F_peak[i%DEFAULT_EDGE_NUM]         # Max FLOPs per node
            base_energy_limit = (P_peak[i%DEFAULT_EDGE_NUM]*(3600)*energy_scale[i%DEFAULT_EDGE_NUM])     # Base energy budget for node i
            f_peak_i = F_peak[i%DEFAULT_EDGE_NUM]
            bw_peak_i = Bw_peak[i%DEFAULT_EDGE_NUM]
            P_peak_i = P_peak[i%DEFAULT_EDGE_NUM]
            FlopsWatt_i = f_peak_i/P_peak_i  
            p_idle_i = round_alpha[i%DEFAULT_EDGE_NUM]*(f_peak_i/FlopsWatt_i)
    
            if scale_param is not None:
                energy_limit = base_energy_limit * scale_param
            else:
                energy_limit = base_energy_limit
            node = EdgeNode(
                id=i,
                flops_capacity=base_flops_capacity,
                energy_limit=energy_limit,
                FlopsWatt= FlopsWatt_i,
                f_peak=f_peak_i,
                bw_peak=bw_peak_i,
                p_idle=p_idle_i,
                models=copy.deepcopy(models),
            )
            edge_nodes.append(node)
            
        return edge_nodes
    
    
    def poisson_arrivals(t0, tn, n):
        lam = n / (tn - t0)
        inter_arrivals = np.random.exponential(1 / lam, n)
        arrivals = t0 + np.cumsum(inter_arrivals)
        return arrivals

        
    def scale_trace(
        trace_times,
        target_lambda,
        time_slot,
        seed=42,
    ):
        ts = np.asarray(trace_times, dtype=float)
        if ts.size == 0:
            return ts, {}
        ts.sort()
        total_req = 0
        T_end = float(ts.max())
        n_bins = int(math.ceil(T_end / time_slot)) + 1
        bin_idx = np.floor(ts / time_slot).astype(int)
    
        N_target = target_lambda
    
        rng = np.random.default_rng(seed)
    
        scaled_chunks = []
        added = 0
        removed = 0
        indices_by_bin = [[] for _ in range(n_bins)]
        for i, b in enumerate(bin_idx):
            indices_by_bin[b].append(i)
    
        for b in range(n_bins):
            idxs = np.array(indices_by_bin[b], dtype=int)
            c = idxs.size
    
            if c > N_target:
                keep_local = rng.choice(c, size=N_target, replace=False)
                keep_idxs = idxs[keep_local]
                scaled_chunks.append(ts[keep_idxs])
                total_req += len(ts[keep_idxs])
                removed += (c - N_target)
    
            elif c < N_target:
                if c > 0:
                    scaled_chunks.append(ts[idxs])
                    total_req += len(ts[idxs])
                need = N_target - c
                if need > 0:
                    t0 = b * time_slot
                    extras = t0 + rng.random(need) * time_slot
                    total_req += len(extras)
                    scaled_chunks.append(extras)
                    added += need
            else:
                if c > 0:
                    scaled_chunks.append(ts[idxs])
                    total_req += len(ts[idxs])
    
        scaled = np.concatenate(scaled_chunks)
        scaled.sort()
    
        debug = {
            "target_per_bin": N_target,
            "n_original": int(ts.size),
            "n_scaled": int(scaled.size),
            "added": int(added),
            "removed": int(removed),
        }
        T_end = float(scaled.max())
        n_bins = int(math.ceil(T_end / time_slot)) + 1
        bin_idx = np.floor(scaled / time_slot).astype(int)
        return scaled, debug


    def load_alibaba_trace_times(max_rows):
        ts = []
        with open("time_stamps_alibaba.csv", newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                try:
                    ts.append(float(row[0]))
                except ValueError:
                    continue
                if len(ts) >= max_rows:
                    break
        if not ts:
            raise ValueError("Alibaba trace is empty or invalid.")
        ts = np.array(ts, dtype=float)
        ts.sort()
        ts = ts - ts[0]
        return ts    

    
    def compute_Pk_8020(models, low_mass=0.8, high_mass=0.2, eps=1e-6):
        acc = np.array([m.accuracy for m in models], dtype=float)
        K = len(models)
        if not np.isclose(low_mass + high_mass, 1.0):
            raise ValueError("low_mass + high_mass must equal 1.0")
    
        mean_acc = float(acc.mean())
    
        low_idx = np.where(acc < mean_acc)[0]
        high_idx = np.where(acc >= mean_acc)[0]       
        Pk = np.zeros(K, dtype=float)
    
        def assign_group(idxs, mass):
            if idxs.size == 0:
                return 0.0
            num_mdl = len(idxs)
            new_pk = [[] for ii in idxs]
            for i in range(len(idxs)):
                new_pk[i] = float(mass*(1/num_mdl))
            return new_pk
        Pk[low_idx] = assign_group(low_idx, low_mass)
        Pk[high_idx] = assign_group(high_idx, high_mass)
        return Pk


    def compute_arrival(edge_num, all_req_num, alpha_peak, Pk_list) -> float:
        C_sys = 0.0
        M_sys = 0.0
        for j in range(edge_num):
            C_sys += ((F_peak[j % DEFAULT_EDGE_NUM]) * (UTILIZATION_THRESHOLD))
            M_sys += Bw_peak[j%DEFAULT_EDGE_NUM]
        E_C = float(sum(float(Pk_list[k]) * float(models[k].required_flops) for k in range(len(models))))
        E_M = float(sum(float(Pk_list[k]) * float(models[k].required_bw) for k in range(len(models))))
        lambda_max = min(C_sys / E_C, M_sys / E_M ) 
        return float(alpha_peak) * float(lambda_max)


    def generate_requestSet(max_rows):
        base_requests = []
        if max_rows is None:
            max_rows = 1000
        trace_times = load_alibaba_trace_times(max_rows)
        rng_attr = np.random.default_rng(2025)
        required_acc_list = [
            round(float(rng_attr.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY)), 2)
            for _ in trace_times
        ]
        qos_rt_list = [
            round(float(rng_attr.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME)), 2)
            for _ in trace_times
        ]
        Pk = compute_Pk_8020(models, 0.8, 0.2, 1e-6)
        lambda_max  = compute_arrival(
            edge_num=DEFAULT_EDGE_NUM,
            all_req_num=len(trace_times),
            alpha_peak=0.95,
            Pk_list=Pk
        )
        lambda_max = math.ceil(lambda_max)
        scaled_arrivals, dbg =  scale_trace(trace_times,lambda_max,TIME_SLOT, seed=42)
        n = len(scaled_arrivals)
        if n > len(required_acc_list):
            extra = n - len(required_acc_list)
            required_acc_list.extend(
                round(float(rng_attr.uniform(MIN_REQUIRED_ACCURACY, MAX_REQUIRED_ACCURACY)), 2)
                for _ in range(extra)
            )
            qos_rt_list.extend(
                round(float(rng_attr.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME)), 2)
                for _ in range(extra)
            )

        req_id = 0
        for t, required_acc, qos_rt in zip(scaled_arrivals, required_acc_list[:n], qos_rt_list[:n]):
            
            FIXED_REQ_NUM = NUM_REQUEST  
            req_num = int(FIXED_REQ_NUM)   # fixed number of requests 
            
            per_req_arrivals = [float(t)] * req_num
    
            rset = Request_set(
                id=req_id,
                arrival_time=float(t),
                qos_accuracy=required_acc,
                qos_response_time=qos_rt,
                num_completed_request=0,
                estimated_accuracy=[],
                finish_time=0,
                flag=False,
            )
            rset.init_requests(per_req_arrivals, req_num) 
            base_requests.append(rset)
            req_id += 1
    
        return base_requests

    base_requests = generate_requestSet(None)

    edge_num = 3 #DEFAULT_EDGE_NUM

    
    time1 = time2 = 0
    time1 = time.time() ;
    sim1=[]
    sim1 = Simulator(edge_nodes=build_nodes(edge_num, None), models=copy.deepcopy(models), duration=60, decision_maker=DecisionMaker_energy_priority())
    sim1.request_sets = copy.deepcopy(base_requests)
    sim1.run()
    log=[]
    time2 = time.time()
    time_spended = time2-time1
    print("time_spended:  ", time_spended)
    approach = "Energy-aware-Model-oriented"
    sim1.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log)
    sim1.save_response_logs(approach)
    sim1.save_response_logs_2(approach)

    

    time1 = time2 = 0
    time1 = time.time() ;
    sim3 = []
    sim3 = Simulator(edge_nodes=build_nodes(edge_num, None), models=copy.deepcopy(models), duration=60, decision_maker = DecisionMaker_edgeParams_request_priority_MlFirst())
    sim3.request_sets = copy.deepcopy(base_requests)
    sim3.run()
    log=[]
    time2 = time.time()
    time_spended = time2-time1
    print("time_spended:  ", time_spended)
    approach = "Multi-criteria-Model-oriented"
    sim3.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log)
    sim3.save_response_logs(approach)
    sim3.save_response_logs_2(approach)

    
    
    time1 = time2 = 0
    time1 = time.time() ;
    sim6 = []
    log=[]
    sim6 = Simulator(edge_nodes=build_nodes(edge_num, None), models=copy.deepcopy(models), duration=60, decision_maker = DecisionMaker_edgeParams_request_priority_reqFirst())
    sim6.request_sets = copy.deepcopy(base_requests)
    sim6.run()
    time2 = time.time()
    time_spended = time2-time1
    print("time_spended:  ", time_spended)
    approach = "Multi-criteria-Request-oriented"
    sim6.print_stats(approach, edge_num, DEFAULT_ARRIVAL, None, log)
    sim6.save_response_logs(approach)
    sim6.save_response_logs_2(approach)


        
if __name__ == "__main__":
    main()


