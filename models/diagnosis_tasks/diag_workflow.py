'''
a workflow for multiple diagnosis tasks
'''
from .anomaly_detection import AD
from diagnosis_tasks.root_cause_localization import RCL
import time 
def diag_workflow(config, time_model,space_model, train_samples, test_samples, 
                  cases, ad_cases_label, 
                  node_dict, type_hash, type_dict, channel_dict=None,
                  workflow=['AD', 'RCL']):
    tmp_res, eval_res = {}, {}
    # anomaly detection
    if 'AD' in workflow:
        print('AD start. |', '*' *100)
        start_time	= time.perf_counter()  # 记录开始时间戳

        tmp_param = config['AD']
        split_ratio = tmp_param['split_ratio']
        method = tmp_param['method']
        t_value = tmp_param['t_value']
        q = tmp_param['q']
        level = tmp_param['level']
        delay = tmp_param['delay']
        impact_window = tmp_param['impact_window']
        verbose = tmp_param['verbose']

        pre_interval, precision, recall, f1 = AD(time_model, train_samples, test_samples, ad_cases_label, 
                                                split_ratio, method, t_value, q, level, delay, impact_window, verbose)
        tmp_res['AD'] = {'pre_interval': pre_interval}
        eval_res['AD'] = {'precision': precision, 'recall': recall, 'f1': f1}
        end_time = time.perf_counter()  # 记录结束时间戳
        print(f"AD耗时 {end_time - start_time:.3f} s")
        print('AD done. | ', eval_res['AD'])

    # root cause localization
    if 'RCL' in workflow:
        print('RCL start. |', '*' *100)
        start_time	= time.perf_counter()  # 记录开始时间戳

        tmp_param = config['RCL']
        split_ratio = tmp_param['split_ratio']
        method = tmp_param['method']
        t_value = tmp_param['t_value']
        before = tmp_param['before']
        after = tmp_param['after']
        verbose = tmp_param['verbose']

        rank_df, topK, avgK = RCL(time_model,space_model, test_samples, cases, node_dict, 
                                split_ratio, method, t_value, before, after, verbose)
        tmp_res['RCL'] = {'rank_df': rank_df}
        eval_res['RCL'] = {'topK': topK, 'avgK': avgK}
        end_time = time.perf_counter()  # 记录结束时间戳
        print(f"RCL耗时 {end_time - start_time:.3f} s")
        print('RCL done. |', eval_res['RCL'])
    return tmp_res, eval_res